"""
MIBA — Supervisor Verification Node
Service: supervisor
WasteKI multi-agent supervisor — verifies work orders via 3-of-4 signal consensus
Signals: GPS track validity · GPS dwell time · Photo match · Weight plausibility
"""

import os
import math
import logging
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from .database import db, connect_to_mongo, close_mongo_connection


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MIBA Supervisor Node",
    description="WasteKI — 3-of-4 consensus work order verification",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*", "https://miba-ui-460918627115.europe-west1.run.app"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_db_client():
    await connect_to_mongo()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_mongo_connection()


TOKEN_ENGINE_URL = os.getenv("TOKEN_ENGINE_URL", "http://token-engine:8080")
CARBON_ENGINE_URL = os.getenv("CARBON_ENGINE_URL", "http://carbon-engine:8080")

# Verification thresholds
MIN_DWELL_SECONDS = int(os.getenv("MIN_DWELL_SECONDS", "120"))      # 2 min minimum at site
MAX_GPS_DRIFT_METRES = float(os.getenv("MAX_GPS_DRIFT_METRES", "50"))  # 50m tolerance
MIN_PHOTO_DIFF_SCORE = float(os.getenv("MIN_PHOTO_DIFF_SCORE", "0.3")) # before vs after must differ ≥30%
CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", "3"))     # 3 of 4 signals must pass

# ─── Database: MongoDB (replacing in-memory Phase 1) ───────────────────────
# Collections: work_orders



# ─── Schemas ───────────────────────────────────────────────────────────────

class GPSPoint(BaseModel):
    lat: float
    lng: float
    timestamp: str
    accuracy_metres: float = 10.0

class WorkOrderCreateRequest(BaseModel):
    work_order_id: Optional[str] = None
    reporter_id: str
    collector_id: Optional[str] = None
    report_lat: float
    report_lng: float
    report_photo_hash: str
    classification_result: dict      # from classification-api response
    image_data: Optional[str] = None  # base64 image for before photo display

class WorkOrderVerifyRequest(BaseModel):
    work_order_id: str
    collector_id: str
    gps_track: list[GPSPoint] = Field(..., min_items=2)
    after_photo_hash: str
    weight_declared_kg: float = Field(..., gt=0)
    before_photo_hash: Optional[str] = None

class VerificationResult(BaseModel):
    work_order_id: str
    verified: bool
    consensus_score: float              # 0.0–1.0
    signals_passed: int
    signals_total: int
    signal_details: dict
    rejection_reason: Optional[str]
    tokens_issued: Optional[dict]
    audit_trail_id: str
    timestamp: str


# ─── GPS utility functions ─────────────────────────────────────────────────

def haversine_metres(lat1, lng1, lat2, lng2) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def validate_gps_track(
    track: list[GPSPoint],
    target_lat: float,
    target_lng: float,
) -> dict:
    """Check if GPS track passes through target location and has sufficient dwell."""
    if len(track) < 2:
        return {"passed": False, "reason": "Insufficient GPS points"}

    # Check proximity: at least one point within MAX_GPS_DRIFT_METRES of target
    distances = [haversine_metres(p.lat, p.lng, target_lat, target_lng) for p in track]
    min_dist = min(distances)
    proximity_ok = min_dist <= MAX_GPS_DRIFT_METRES

    # Dwell time: time spent within 100m of target
    near_points = [p for p, d in zip(track, distances) if d <= 100]
    dwell_seconds = 0
    if len(near_points) >= 2:
        try:
            t_start = datetime.fromisoformat(near_points[0].timestamp)
            t_end = datetime.fromisoformat(near_points[-1].timestamp)
            dwell_seconds = (t_end - t_start).total_seconds()
        except Exception:
            dwell_seconds = 0

    dwell_ok = dwell_seconds >= MIN_DWELL_SECONDS

    # Spoofing detection: check for teleportation (speed > 200 km/h between points)
    spoofing_detected = False
    for i in range(1, len(track)):
        d = haversine_metres(track[i-1].lat, track[i-1].lng, track[i].lat, track[i].lng)
        try:
            dt = (datetime.fromisoformat(track[i].timestamp) -
                  datetime.fromisoformat(track[i-1].timestamp)).total_seconds()
            if dt > 0 and (d / dt) > 55:  # 55 m/s = 200 km/h
                spoofing_detected = True
                break
        except Exception:
            pass

    passed = proximity_ok and dwell_ok and not spoofing_detected
    return {
        "passed": passed,
        "proximity_metres": round(min_dist, 1),
        "dwell_seconds": round(dwell_seconds, 1),
        "spoofing_detected": spoofing_detected,
        "reason": None if passed else (
            "GPS spoofing detected" if spoofing_detected else
            f"Collector did not approach within {MAX_GPS_DRIFT_METRES}m" if not proximity_ok else
            f"Dwell time {dwell_seconds:.0f}s < minimum {MIN_DWELL_SECONDS}s"
        ),
    }


def validate_photo_diff(before_hash: Optional[str], after_hash: str) -> dict:
    """
    Production: compare perceptual hash similarity between before/after photos.
    Phase 1 stub: pass if hashes are different (basic check).
    """
    if before_hash is None:
        return {"passed": True, "score": 0.5, "note": "No before photo — soft pass"}
    if before_hash == after_hash:
        return {"passed": False, "score": 0.0, "reason": "Before and after photo are identical"}
    # TODO Phase 2: use imagehash library for perceptual diff score
    # score = 1 - (imagehash.phash(before) - imagehash.phash(after)) / 64
    return {"passed": True, "score": 0.7, "note": "Hash-diff check passed (perceptual diff in Phase 2)"}


def validate_weight_plausibility(
    weight_kg: float,
    classification: dict,
) -> dict:
    """Check declared weight is plausible given the classification volume estimate."""
    estimated_kg = classification.get("total_weight_kg_estimate", 0)
    if estimated_kg <= 0:
        return {"passed": True, "score": 0.5, "note": "No weight estimate available"}

    ratio = weight_kg / estimated_kg if estimated_kg > 0 else 0
    passed = 0.2 <= ratio <= 5.0  # Allow 5x variance (rough estimate)

    return {
        "passed": passed,
        "declared_kg": weight_kg,
        "estimated_kg": round(estimated_kg, 3),
        "ratio": round(ratio, 2),
        "reason": None if passed else f"Declared weight {weight_kg}kg is {ratio:.1f}× estimated — suspicious",
    }


# ─── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    count = await db.db["work_orders"].count_documents({"status": "OPEN"})
    return {"status": "ok", "service": "supervisor", "open_work_orders": count}



@app.post("/workorder/create")
async def create_work_order(req: WorkOrderCreateRequest):
    """Create a new work order from a citizen report."""
    wo_id = req.work_order_id or f"WO-{str(uuid.uuid4())[:8].upper()}"
    wo_data = {
        "work_order_id": wo_id,
        "status": "OPEN",
        "reporter_id": req.reporter_id,
        "collector_id": req.collector_id,
        "location": {"lat": req.report_lat, "lng": req.report_lng},
        "before_photo_hash": req.report_photo_hash,
        "image_data": req.image_data,
        "classification": req.classification_result,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    await db.db["work_orders"].insert_one(wo_data)
    logger.info("Work order created: %s at (%.4f, %.4f)", wo_id, req.report_lat, req.report_lng)
    return {"work_order_id": wo_id, "status": "OPEN", "message": "Work order created"}



@app.post("/workorder/verify", response_model=VerificationResult)
async def verify_work_order(req: WorkOrderVerifyRequest, bg: BackgroundTasks):
    """Run 3-of-4 consensus verification and optionally issue tokens."""
    wo = await db.db["work_orders"].find_one({"work_order_id": req.work_order_id})
    if not wo:
        raise HTTPException(404, f"Work order {req.work_order_id} not found")
    if wo["status"] == "CLOSED":
        raise HTTPException(409, "Work order already closed")


    location = wo["location"]
    classification = wo.get("classification", {})

    # ── Run all 4 signals ──────────────────────────────────────────────────
    gps_result = validate_gps_track(req.gps_track, location["lat"], location["lng"])
    dwell_result = {
        "passed": gps_result.get("dwell_seconds", 0) >= MIN_DWELL_SECONDS,
        "dwell_seconds": gps_result.get("dwell_seconds", 0),
    }
    photo_result = validate_photo_diff(wo.get("before_photo_hash"), req.after_photo_hash)
    weight_result = validate_weight_plausibility(req.weight_declared_kg, classification)

    signals = {
        "gps_proximity": {"passed": gps_result["passed"], "detail": gps_result},
        "gps_dwell":     {"passed": dwell_result["passed"], "detail": dwell_result},
        "photo_diff":    {"passed": photo_result["passed"], "detail": photo_result},
        "weight_plausibility": {"passed": weight_result["passed"], "detail": weight_result},
    }

    passed_count = sum(1 for s in signals.values() if s["passed"])
    consensus_score = round(passed_count / 4, 2)
    verified = passed_count >= CONSENSUS_THRESHOLD

    rejection_reason = None
    if not verified:
        failed = [k for k, v in signals.items() if not v["passed"]]
        rejection_reason = f"Consensus failed: {passed_count}/4 signals passed. Failed: {', '.join(failed)}"

    # ── Update work order ──────────────────────────────────────────────────
    update_data = {
        "status": "CLOSED" if verified else "DISPUTED",
        "collector_id": req.collector_id,
        "verification": {
            "verified": verified,
            "consensus_score": consensus_score,
            "signals": signals,
            "verified_at": datetime.now(timezone.utc).isoformat(),
        },
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
    await db.db["work_orders"].update_one(
        {"work_order_id": req.work_order_id},
        {"$set": update_data}
    )


    audit_id = str(uuid.uuid4())

    # ── Trigger token issuance if verified ────────────────────────────────
    tokens_issued = None
    if verified:
        bg.add_task(trigger_token_issuance, req.work_order_id, req.collector_id,
                    classification, req.weight_declared_kg, consensus_score)

    logger.info("WO %s verified=%s score=%.2f signals=%d/4",
                req.work_order_id, verified, consensus_score, passed_count)

    return VerificationResult(
        work_order_id=req.work_order_id,
        verified=verified,
        consensus_score=consensus_score,
        signals_passed=passed_count,
        signals_total=4,
        signal_details=signals,
        rejection_reason=rejection_reason,
        tokens_issued=tokens_issued,
        audit_trail_id=audit_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def trigger_token_issuance(wo_id, collector_id, classification, weight_kg, score):
    """Background task: call token-engine to issue tokens after verification."""
    try:
        dominant = classification.get("dominant_category", "W16")
        co2e = classification.get("total_co2e_avoided_kg", 0.0)
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(f"{TOKEN_ENGINE_URL}/tokens/issue", json={
                "work_order_id": wo_id,
                "collector_id": collector_id,
                "category_code": dominant,
                "weight_kg": weight_kg,
                "co2e_avoided_kg": co2e,
                "supervisor_verified": True,
                "verification_score": score,
            })
            if resp.status_code == 200:
                logger.info("Tokens issued for WO %s: %s", wo_id, resp.json())
    except Exception as e:
        logger.error("Token issuance failed for WO %s: %s", wo_id, e)


@app.get("/workorder/{work_order_id}")
async def get_work_order(work_order_id: str):
    wo = await db.db["work_orders"].find_one({"work_order_id": work_order_id})
    if not wo:
        raise HTTPException(404, "Work order not found")
    # Remove MongoDB internal _id if present for JSON serialization
    if "_id" in wo:
        del wo["_id"]
    return wo


@app.get("/workorder")
async def list_work_orders(status: Optional[str] = None, limit: int = 50):
    query = {}
    if status:
        query["status"] = status.upper()
    
    cursor = db.db["work_orders"].find(query).sort("created_at", -1).limit(limit)
    orders = await cursor.to_list(length=limit)
    
    for o in orders:
        if "_id" in o:
            del o["_id"]
            
    return {"total": len(orders), "work_orders": orders}


# ─── Multi-Agent Analytics ────────────────────────────────────────────────

@app.get("/ai/analytics")
async def ai_analytics():
    """Aggregated stats for the UI Dashboard."""
    # Run multiple aggregations in parallel for efficiency using a single $facet query
    fourteen_days_ago = datetime.now(timezone.utc) - timedelta(days=14)

    main_pipeline = [
        {
            "$facet": {
                "totals": [
                    {
                        "$group": {
                            "_id": None,
                            "total_reports": {"$sum": 1},
                            "closed_reports": {"$sum": {"$cond": [{"$eq": ["$status", "CLOSED"]}, 1, 0]}},
                            "total_weight": {"$sum": "$classification.total_weight_kg_estimate"},
                            "total_co2": {"$sum": "$classification.total_co2e_avoided_kg"},
                            "high_sev": {"$sum": {"$cond": [{"$gt": ["$classification.total_weight_kg_estimate", 5]}, 1, 0]}},
                            "medium_sev": {"$sum": {"$cond": [{"$and": [{"$gt": ["$classification.total_weight_kg_estimate", 1]}, {"$lte": ["$classification.total_weight_kg_estimate", 5]}]}, 1, 0]}},
                            "low_sev": {"$sum": {"$cond": [{"$lte": ["$classification.total_weight_kg_estimate", 1]}, 1, 0]}},
                        }
                    }
                ],
                "categories": [
                    {"$match": {"classification.dominant_category_name": {"$ne": None}}},
                    {"$group": {"_id": "$classification.dominant_category_name", "count": {"$sum": 1}}}
                ],
                "cities": [
                    {"$match": {"city": {"$ne": None}}},
                    {"$group": {"_id": "$city", "count": {"$sum": 1}}}
                ],
                "timeline_counts": [
                    {"$match": {"created_at": {"$gte": fourteen_days_ago.isoformat()}}},
                    {"$group": {"_id": {"$substr": ["$created_at", 0, 10]}, "count": {"$sum": 1}}}
                ]
            }
        }
    ]

    agg_result = await db.db["work_orders"].aggregate(main_pipeline).to_list(1)
    if not agg_result: # No data in collection
        return {"summary": {"total": 0}, "agentReports": {}, "aiInsights": "No data available."}

    data = agg_result[0]
    summary_totals = data["totals"][0] if data["totals"] else {}
    total_reports = summary_totals.get("total_reports", 0)
    closed_reports = summary_totals.get("closed_reports", 0)
    total_weight = summary_totals.get("total_weight", 0.0)
    total_co2 = summary_totals.get("total_co2", 0.0)

    categories = {c["_id"]: c["count"] for c in data["categories"] if c["_id"]}
    cities = {c["_id"]: c["count"] for c in data["cities"] if c["_id"]}
    severities = {"High": summary_totals.get("high_sev", 0), "Medium": summary_totals.get("medium_sev", 0), "Low": summary_totals.get("low_sev", 0)}

    # Create and populate timeline for the last 14 days
    timeline = []
    timeline_db_counts = {item["_id"]: item["count"] for item in data["timeline_counts"]}
    now = datetime.now(timezone.utc)
    for i in range(13, -1, -1):
        day = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        timeline.append({"date": day, "count": timeline_db_counts.get(day, 0)})

    return {
        "summary": {
            "total": total_reports,
            "totalWeight": round(total_weight, 1),
            "totalCo2": round(total_co2, 1),
            "cleanedPercentage": round((closed_reports / total_reports * 100) if total_reports > 0 else 0, 1),
            "cleanedCount": closed_reports,
            "timeline": timeline,
            "categories": categories,
            "severities": severities,
            "cities": cities
        },
        "agentReports": {
            "trend": "up" if total_reports > 0 else "stable",
            "severity": max(severities, key=severities.get) if total_reports > 0 else "Low",
            "city": max(cities, key=cities.get) if cities else "N/A"
        },
        "aiInsights": f"MIBA detected {total_reports} waste incidents. {closed_reports} have been verified and cleared. Dominant waste type: {max(categories, key=categories.get) if categories else 'N/A'}."
    }


@app.post("/ai/analytics/chat")
async def ai_chat(payload: dict):
    """Simple RAG placeholder using Vertex AI (if configured) or static response."""
    user_msg = payload.get("message", "").lower()
    
    # In a real scenario, we'd use Vertex AI Gemini here.
    # For local dev, we provide a smart response based on DB stats.
    stats = await ai_analytics()
    total = stats["summary"]["total"]
    
    if "how many" in user_msg or "total" in user_msg:
        response = f"There are currently {total} total reports in the system."
    elif "weight" in user_msg:
        response = f"The total estimated weight of waste detected is {stats['summary']['totalWeight']} kg."
    else:
        response = "I am the MIBA AI Assistant. I can help you analyze waste data. Current system status: Healthy."

    return {"response": response}


@app.get("/supervisor/stats")
async def stats():
    total = await db.db["work_orders"].count_documents({})
    open_count = await db.db["work_orders"].count_documents({"status": "OPEN"})
    closed = await db.db["work_orders"].count_documents({"status": "CLOSED"})
    disputed = await db.db["work_orders"].count_documents({"status": "DISPUTED"})
    
    return {
        "total_work_orders": total,
        "open": open_count,
        "closed_verified": closed,
        "disputed": disputed,
        "consensus_threshold": CONSENSUS_THRESHOLD,
    }
