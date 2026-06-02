"""
FailSight RAG Agent
LangChain + FAISS over historical maintenance logs

Field engineer asks: "What failed last time Chamber 3 showed
this pressure spike?" — agent retrieves relevant maintenance
history and generates a grounded diagnostic recommendation.

This is Gap 2 fix: embedding-based search over domain text.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Optional
from datetime import datetime, timedelta
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path("data/embeddings/")
EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# ── Maintenance Log Generation ────────────────────────────────────────────────

def generate_maintenance_logs() -> List[Document]:
    """
    Generate realistic semiconductor fab maintenance logs.
    These are the kinds of records field engineers at Applied Materials,
    KLA, Lam Research actually write after servicing equipment.

    In production: these come from a CMMS (computerized maintenance
    management system) like SAP PM or IBM Maximo.
    """
    random.seed(42)

    equipment_types = [
        "CVD Chamber", "Etch System", "Ion Implanter",
        "CMP Tool", "Lithography System", "Metrology Station",
        "Thermal Furnace", "Wafer Handler",
    ]

    failure_modes = {
        "CVD Chamber": [
            ("pressure_spike", "Pressure excursion above setpoint. Found cracked O-ring on gas inlet manifold. Replaced O-ring, leak-checked to spec. Root cause: thermal cycling fatigue after 2400 hours."),
            ("temperature_drift", "Heater zone 3 drifting +15°C from setpoint. Thermocouple degraded. Replaced TC assembly. Verified uniformity within ±2°C across wafer."),
            ("particle_event", "High particle count on monitor wafer. Identified delamination on chamber wall coating. Performed full chamber clean, recoat, and seasoning."),
            ("rf_power_instability", "RF power fluctuating ±8% during deposition. Matching network capacitor degraded. Replaced capacitor bank. Process restored to spec."),
        ],
        "Etch System": [
            ("pressure_spike", "Sudden pressure rise during etch step. Throttle valve actuator stuck. Replaced actuator and recalibrated pressure control loop."),
            ("endpoint_drift", "OES endpoint detection missing etch completion by 12 seconds. Recalibrated OES wavelength window for SiO2 endpoint signature."),
            ("temperature_drift", "Chuck temperature unstable. Coolant flow sensor failed. Replaced sensor, verified chuck temperature control to ±0.5°C."),
            ("rf_power_instability", "Plasma instability during high-aspect-ratio etch. ESC voltage drifting. Replaced ESC power supply filter capacitors."),
        ],
        "Ion Implanter": [
            ("beam_current_drop", "Beam current dropped 40% during implant. Ion source filament near end of life. Replaced filament assembly. Beam current restored within spec."),
            ("vacuum_excursion", "Vacuum pressure spike in end station. Gate valve seat worn. Replaced valve seat assembly and leak-checked."),
            ("dose_uniformity", "Dose uniformity degraded to ±3.2% (spec ±2%). Scan motor encoder drift. Replaced encoder, recalibrated scan uniformity."),
        ],
        "CMP Tool": [
            ("pressure_spike", "Wafer carrier pressure unstable. Membrane rupture on carrier head 2. Replaced membrane, rebuilt carrier, qualified with monitor wafers."),
            ("temperature_drift", "Slurry temperature rising above target. Heat exchanger fouled. Cleaned HX, restored slurry temp to 22±0.5°C."),
            ("particle_event", "Scratch defects on product wafers. Pad conditioning disk worn. Replaced conditioner disk, ran pad break-in, qualified."),
        ],
    }

    logs = []
    base_date = datetime.now() - timedelta(days=365)

    for i in range(200):
        equip = random.choice(equipment_types)
        chamber_id = f"{equip[:3].upper()}-{random.randint(1,6):02d}"
        event_date = base_date + timedelta(days=random.randint(0, 365))

        if equip in failure_modes:
            failure_type, description = random.choice(failure_modes[equip])
        else:
            failure_type = "general_maintenance"
            description = f"Scheduled PM performed on {equip}. All parameters within spec post-maintenance."

        # Sensor readings at time of failure (realistic ranges)
        pressure = round(random.uniform(1.2, 8.5), 3)
        temperature = round(random.uniform(620, 680), 1)
        rf_power = round(random.uniform(480, 520), 1)
        cycles_since_pm = random.randint(200, 3000)

        log_text = f"""
MAINTENANCE LOG — {chamber_id}
Date: {event_date.strftime('%Y-%m-%d')}
Equipment: {equip}
Failure Mode: {failure_type}
Cycles Since Last PM: {cycles_since_pm}

Sensor Readings at Event:
  Chamber Pressure: {pressure} Torr
  Process Temperature: {temperature} °C
  RF Power: {rf_power} W

Description: {description}

Downtime: {random.randint(2, 24)} hours
Parts Replaced: {'Yes' if random.random() > 0.4 else 'No'}
Engineer Sign-off: Field Service Engineer {random.choice(['A. Kumar', 'J. Park', 'M. Chen', 'R. Santos'])}
        """.strip()

        logs.append(Document(
            page_content=log_text,
            metadata={
                "chamber_id": chamber_id,
                "equipment_type": equip,
                "failure_mode": failure_type,
                "date": event_date.strftime("%Y-%m-%d"),
                "source": "maintenance_log",
            },
        ))

    # Add equipment health summaries
    for equip in equipment_types:
        summary = f"""
EQUIPMENT HEALTH SUMMARY — {equip}
Most Common Failure Modes: {', '.join([fm[0] for fm in failure_modes.get(equip, [('general_maintenance', '')])])}
Average Cycles Between Failures: {random.randint(800, 2500)}
Average Downtime Per Event: {random.randint(4, 18)} hours
Highest Risk Sensor: {'Chamber Pressure' if 'pressure' in str(failure_modes.get(equip, '')) else 'Temperature'}
Recommended PM Interval: {random.randint(500, 1000)} cycles
        """.strip()

        logs.append(Document(
            page_content=summary,
            metadata={"equipment_type": equip, "source": "health_summary"},
        ))

    logger.info(f"Generated {len(logs)} maintenance log documents")
    return logs


def build_vector_store(documents: List[Document]) -> FAISS:
    """Embed maintenance logs and store in FAISS."""
    logger.info("Building FAISS index over maintenance logs...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(EMBEDDINGS_PATH / "maintenance_logs_index"))
    logger.info("FAISS index saved")
    return vectorstore


def load_vector_store() -> Optional[FAISS]:
    index_path = EMBEDDINGS_PATH / "maintenance_logs_index"
    if not index_path.exists():
        return None
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    return FAISS.load_local(
        str(index_path), embeddings, allow_dangerous_deserialization=True
    )


def build_rag_chain(vectorstore: FAISS):
    """Build LangChain RAG chain for maintenance Q&A."""
    from langchain_community.llms import HuggingFaceHub

    prompt_template = """
You are FailSight, an AI assistant for semiconductor fab field engineers.
Use the maintenance log history below to answer the engineer's question.
Be specific: reference equipment IDs, failure modes, sensor readings, and actions taken.
If the history doesn't contain relevant information, say so clearly.

Maintenance History Context:
{context}

Engineer Question: {question}

Diagnostic Answer:
"""

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={"temperature": 0.1, "max_length": 512},
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )
    return chain


def query_agent(chain, question: str) -> dict:
    result = chain({"query": question})
    return {
        "question": question,
        "answer": result["result"],
        "sources": [
            {
                "chamber": doc.metadata.get("chamber_id", "N/A"),
                "date": doc.metadata.get("date", "N/A"),
                "failure_mode": doc.metadata.get("failure_mode", "N/A"),
                "excerpt": doc.page_content[:200],
            }
            for doc in result["source_documents"]
        ],
    }


if __name__ == "__main__":
    logs = generate_maintenance_logs()
    vectorstore = build_vector_store(logs)
    chain = build_rag_chain(vectorstore)

    questions = [
        "What failed last time a CVD chamber showed a pressure spike?",
        "What is the average downtime for etch system failures?",
        "Which failure mode causes the most frequent ion implanter downtime?",
        "What sensor readings preceded the last O-ring failure?",
    ]

    print("\n" + "=" * 60)
    print("FAILSIGHT MAINTENANCE LOG AGENT")
    print("=" * 60)

    for q in questions:
        result = query_agent(chain, q)
        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print("-" * 40)
