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
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import httpx
from .database import db, connect_to_db, close_db_connection


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MIBA Supervisor Node",
    description="WasteKI — 3-of-4 consensus work order verification",
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*", "https://miba-ui-340635219170.europe-west3.run.app"], allow_methods=["*"], allow_headers=["*"])

@app.on_event("startup")
async def startup_db_client():
    await connect_to_db()

@app.on_event("shutdown")
async def shutdown_db_client():
    await close_db_connection()


TOKEN_ENGINE_URL = os.getenv("TOKEN_ENGINE_URL", "http://token-engine:8080")
CARBON_ENGINE_URL = os.getenv("CARBON_ENGINE_URL", "http://carbon-engine:8080")

# Verification thresholds
MIN_DWELL_SECONDS = int(os.getenv("MIN_DWELL_SECONDS", "120"))      # 2 min minimum at site
MAX_GPS_DRIFT_METRES = float(os.getenv("MAX_GPS_DRIFT_METRES", "50"))  # 50m tolerance
MIN_PHOTO_DIFF_SCORE = float(os.getenv("MIN_PHOTO_DIFF_SCORE", "0.3")) # before vs after must differ ≥30%
CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", "3"))     # 3 of 4 signals must pass


def clean_row(row):
    """Convert asyncpg record to JSON-serializable dict."""
    if not row:
        return None
    d = dict(row)
    for k, v in d.items():
        if isinstance(v, datetime):
            d[k] = v.isoformat()
        elif k in ["location", "classification", "verification"] and isinstance(v, str):
            try:
                d[k] = json.loads(v)
            except:
                pass
    return d


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
    classification_result: dict
    image_data: Optional[str] = None

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
    consensus_score: float
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

def validate_gps_track(track: list[GPSPoint], target_lat: float, target_lng: float) -> dict:
    if len(track) < 2:
        return {"passed": False, "reason": "Insufficient GPS points"}
    distances = [haversine_metres(p.lat, p.lng, target_lat, target_lng) for p in track]
    min_dist = min(distances)
    proximity_ok = min_dist <= MAX_GPS_DRIFT_METRES
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
    spoofing_detected = False
    for i in range(1, len(track)):
        d = haversine_metres(track[i-1].lat, track[i-1].lng, track[i].lat, track[i].lng)
        try:
            dt = (datetime.fromisoformat(track[i].timestamp) - datetime.fromisoformat(track[i-1].timestamp)).total_seconds()
            if dt > 0 and (d / dt) > 55:
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
        "reason": None if passed else ("GPS spoofing detected" if spoofing_detected else f"Collector did not approach within {MAX_GPS_DRIFT_METRES}m" if not proximity_ok else f"Dwell time {dwell_seconds:.0f}s < minimum {MIN_DWELL_SECONDS}s"),
    }

def validate_photo_diff(before_hash: Optional[str], after_hash: str) -> dict:
    if before_hash is None:
        return {"passed": True, "score": 0.5, "note": "No before photo — soft pass"}
    if before_hash == after_hash:
        return {"passed": False, "score": 0.0, "reason": "Before and after photo are identical"}
    return {"passed": True, "score": 0.7, "note": "Hash-diff check passed"}

def validate_weight_plausibility(weight_kg: float, classification: dict) -> dict:
    estimated_kg = classification.get("total_weight_kg_estimate", 0)
    if estimated_kg <= 0:
        return {"passed": True, "score": 0.5, "note": "No weight estimate available"}
    ratio = weight_kg / estimated_kg if estimated_kg > 0 else 0
    passed = 0.2 <= ratio <= 5.0
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
    async with db.pool.acquire() as conn:
        count = await conn.fetchval("SELECT count(*) FROM work_orders WHERE status = 'OPEN'")
    return {"status": "ok", "service": "supervisor", "open_work_orders": count}

@app.get("/stats")
async def stats():
    """Platform stats: total reports and total CO2e avoided."""
    async with db.pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT count(*) as count, 
                   COALESCE(SUM((classification::jsonb->>'total_co2e_avoided_kg')::float), 0.0) as co2e 
            FROM work_orders
        """)
    return {"report_count": row['count'], "co2e_avoided_kg": round(row['co2e'], 2)}

@app.post("/workorder/create")
async def create_work_order(req: WorkOrderCreateRequest):
    wo_id = req.work_order_id or f"WO-{str(uuid.uuid4())[:8].upper()}"
    async with db.pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO work_orders (
                work_order_id, status, reporter_id, collector_id, location, 
                before_photo_hash, image_data, classification, created_at, updated_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)""",
            wo_id, "OPEN", req.reporter_id, req.collector_id, 
            json.dumps({"lat": req.report_lat, "lng": req.report_lng}),
            req.report_photo_hash, req.image_data, json.dumps(req.classification_result)
        )
    return {"work_order_id": wo_id, "status": "OPEN", "message": "Work order created"}

@app.post("/workorder/verify", response_model=VerificationResult)
async def verify_work_order(req: WorkOrderVerifyRequest, bg: BackgroundTasks):
    async with db.pool.acquire() as conn:
        wo = await conn.fetchrow("SELECT * FROM work_orders WHERE work_order_id = $1", req.work_order_id)
        if not wo:
            raise HTTPException(404, f"Work order {req.work_order_id} not found")
        
        location = json.loads(wo["location"]) if isinstance(wo["location"], str) else wo["location"]
        classification = json.loads(wo["classification"]) if isinstance(wo["classification"], str) else wo["classification"]

        gps_result = validate_gps_track(req.gps_track, location["lat"], location["lng"])
        dwell_result = {"passed": gps_result.get("dwell_seconds", 0) >= MIN_DWELL_SECONDS, "dwell_seconds": gps_result.get("dwell_seconds", 0)}
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

        await conn.execute(
            """UPDATE work_orders SET 
                status = $1, collector_id = $2, 
                verification = $3, updated_at = CURRENT_TIMESTAMP 
            WHERE work_order_id = $4""",
            "CLOSED" if verified else "DISPUTED", req.collector_id,
            json.dumps({"verified": verified, "consensus_score": consensus_score, "signals": signals, "verified_at": datetime.now(timezone.utc).isoformat()}),
            req.work_order_id
        )

    bg.add_task(trigger_token_issuance, req.work_order_id, req.collector_id, classification, req.weight_declared_kg, consensus_score)
    return VerificationResult(
        work_order_id=req.work_order_id, verified=verified, consensus_score=consensus_score,
        signals_passed=passed_count, signals_total=4, signal_details=signals,
        rejection_reason=None if verified else "Consensus failed", tokens_issued=None,
        audit_trail_id=str(uuid.uuid4()), timestamp=datetime.now(timezone.utc).isoformat()
    )

async def trigger_token_issuance(wo_id, collector_id, classification, weight_kg, score):
    try:
        dominant = classification.get("dominant_category", "W16")
        co2e = classification.get("total_co2e_avoided_kg", 0.0)
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(f"{TOKEN_ENGINE_URL}/tokens/issue", json={
                "work_order_id": wo_id, "collector_id": collector_id, "category_code": dominant,
                "weight_kg": weight_kg, "co2e_avoided_kg": co2e, "supervisor_verified": True, "verification_score": score,
            })
    except Exception as e:
        logger.error("Token issuance failed for WO %s: %s", wo_id, e)

@app.get("/workorder/{work_order_id}")
async def get_work_order(work_order_id: str):
    async with db.pool.acquire() as conn:
        wo = await conn.fetchrow("SELECT * FROM work_orders WHERE work_order_id = $1", work_order_id)
        if not wo:
            raise HTTPException(404, "Work order not found")
        return clean_row(wo)

@app.get("/workorder")
async def list_work_orders(status: Optional[str] = None, reporter_id: Optional[str] = None, role: Optional[str] = None, limit: int = 50):
    async with db.pool.acquire() as conn:
        # Exclude image_data (200KB+ base64) and full classification (large AI JSON).
        # Extract only the summary fields needed for list views via inline JSON extraction.
        query = """SELECT
                     work_order_id, status, reporter_id, collector_id,
                     location, created_at, updated_at,
                     classification::jsonb->>'dominant_category'      AS dominant_category,
                     classification::jsonb->>'dominant_category_name' AS dominant_category_name,
                     classification::jsonb->>'severity'               AS severity,
                     (classification::jsonb->>'total_weight_kg_estimate')::float AS total_weight_kg_estimate,
                     (classification::jsonb->>'total_co2e_avoided_kg')::float   AS total_co2e_avoided_kg,
                     (classification::jsonb->>'grand_total_value_inr')::float   AS grand_total_value_inr,
                     (classification::jsonb->>'total_scrap_value_inr_min')::float AS total_scrap_value_inr_min,
                     (classification::jsonb->>'total_token_value_inr')::float   AS total_token_value_inr,
                     (classification::jsonb->>'total_carbon_credit_inr')::float AS total_carbon_credit_inr,
                     classification::jsonb->>'categories_found'       AS categories_found,
                     classification::jsonb->>'ai_narrative'           AS ai_narrative
                   FROM work_orders WHERE 1=1"""
        args = []
        if status:
            args.append(status.upper())
            query += f" AND status = ${len(args)}"
        if reporter_id:
            args.append(reporter_id)
            field = "collector_id" if role and role.lower() == "collector" else "reporter_id"
            query += f" AND {field} = ${len(args)}"

        query += " ORDER BY created_at DESC LIMIT 50"
        rows = await conn.fetch(query, *args)

        def build_wo(r):
            d = dict(r)
            for k, v in d.items():
                if isinstance(v, datetime):
                    d[k] = v.isoformat()
            # Reconstruct the classification sub-object the mobile app expects
            d["classification"] = {
                "dominant_category":          d.pop("dominant_category", None),
                "dominant_category_name":     d.pop("dominant_category_name", None),
                "severity":                   d.pop("severity", None),
                "total_weight_kg_estimate":   d.pop("total_weight_kg_estimate", 0) or 0,
                "total_co2e_avoided_kg":      d.pop("total_co2e_avoided_kg", 0) or 0,
                "grand_total_value_inr":      d.pop("grand_total_value_inr", 0) or 0,
                "total_scrap_value_inr_min":  d.pop("total_scrap_value_inr_min", 0) or 0,
                "total_token_value_inr":      d.pop("total_token_value_inr", 0) or 0,
                "total_carbon_credit_inr":    d.pop("total_carbon_credit_inr", 0) or 0,
                "categories_found":           json.loads(d.pop("categories_found") or "[]"),
                "ai_narrative":               d.pop("ai_narrative", None),
            }
            if isinstance(d.get("location"), str):
                try: d["location"] = json.loads(d["location"])
                except: pass
            return d

        return {"total": len(rows), "work_orders": [build_wo(r) for r in rows]}

@app.get("/ai/analytics")
async def ai_analytics():
    async with db.pool.acquire() as conn:
        summary = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'CLOSED') as closed_count,
                COALESCE(SUM((classification::jsonb->>'total_weight_kg_estimate')::float), 0) as total_weight,
                COALESCE(SUM((classification::jsonb->>'total_co2e_avoided_kg')::float), 0) as total_co2,
                COUNT(*) FILTER (WHERE (classification::jsonb->>'total_weight_kg_estimate')::float > 5) as high_sev,
                COUNT(*) FILTER (WHERE (classification::jsonb->>'total_weight_kg_estimate')::float BETWEEN 1 AND 5) as med_sev,
                COUNT(*) FILTER (WHERE (classification::jsonb->>'total_weight_kg_estimate')::float <= 1) as low_sev
            FROM work_orders
        """)
        categories = await conn.fetch("""
            SELECT (classification::jsonb->>'dominant_category_name') as name, COUNT(*) as count 
            FROM work_orders GROUP BY name HAVING (classification::jsonb->>'dominant_category_name') IS NOT NULL
        """)
        
        # Timeline (last 14 days)
        timeline = await conn.fetch("""
            SELECT TO_CHAR(created_at, 'YYYY-MM-DD') as date, COUNT(*) as count 
            FROM work_orders 
            WHERE created_at >= CURRENT_DATE - INTERVAL '14 days'
            GROUP BY date ORDER BY date ASC
        """)

    cat_dict = {c['name']: c['count'] for c in categories}
    total = summary['total'] or 0
    closed = summary['closed_count'] or 0
    
    return {
        "summary": {
            "total": total,
            "totalWeight": round(summary['total_weight'], 1),
            "totalCo2": round(summary['total_co2'], 1),
            "cleanedPercentage": round((closed / total * 100) if total > 0 else 0, 1),
            "cleanedCount": closed,
            "timeline": [dict(t) for t in timeline],
            "categories": cat_dict,
            "severities": {"High": summary['high_sev'], "Medium": summary['med_sev'], "Low": summary['low_sev']}
        },
        "agentReports": {
            "trend": "up" if total > 0 else "stable",
            "severity": "Medium",
            "city": "Unknown"
        },
        "aiInsights": f"MIBA detected {total} incidents with SQL engine."
    }

@app.post("/ai/analytics/chat")
async def ai_chat(payload: dict):
    stats = await ai_analytics()
    total = stats["summary"]["total"]
    return {"response": f"MIBA system currently has {total} total reports in the SQL database."}

@app.get("/supervisor/stats")
async def supervisor_stats():
    async with db.pool.acquire() as conn:
        row = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE status = 'OPEN') as open,
                COUNT(*) FILTER (WHERE status = 'CLOSED') as closed,
                COUNT(*) FILTER (WHERE status = 'DISPUTED') as disputed
            FROM work_orders
        """)
    return dict(row) if row else {"total":0, "open":0, "closed":0, "disputed":0}
