"""Local results dashboard: streamlit run dashboard.py."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from metrics import read_episodes, summarize


def display_name(name):
    return {"final_1m_g4_seed1": "Agent · seed 1",
            "final_1m_g4_seed2": "Agent · seed 2",
            "random_seed7001": "Aléatoire"}.get(name, name.replace("_", " "))


st.set_page_config(page_title="Crafter · Agent RL", page_icon="🌱", layout="wide")
st.title("🌱 Crafter · apprentissage par renforcement")
st.caption("Résultats du projet de Kilian Calmets et Mathis Pezet · entraînement, compétences et comparaison à l'aléatoire")
root = Path("runs")
report_path = root / "final_report.json"
random_path = root / "random_seed7001" / "summary.json"
if report_path.exists() and random_path.exists():
    report = json.loads(report_path.read_text(encoding="utf-8"))
    random_report = json.loads(random_path.read_text(encoding="utf-8"))
    col1, col2, col3 = st.columns(3)
    col1.metric("Reward moyen · agent", f"{report['reward_mean']:.2f}",
                delta=f"+{report['reward_mean']-random_report['reward_mean']:.2f} vs aléatoire")
    col2.metric("Score Crafter · agent", f"{report['score_mean']:.2f} %",
                delta=f"+{report['score_mean']-random_report['score']:.2f} points")
    col3.metric("Protocole final", f"{report['training_seeds']} entraînements",
                help="50 nouveaux épisodes par entraînement, comparés aux mêmes épisodes aléatoires")
    st.caption("Checkpoint à 1 000 000 de pas · politique gloutonne (ε = 0) · 50 épisodes par entraînement · ± dans les notes = écart-type entre les 2 entraînements")
with st.sidebar:
    st.header("Tester un agent")
    checkpoints = sorted(root.glob("**/checkpoint_*.pt")) if root.exists() else []
    if checkpoints:
        preferred_checkpoint = next((i for i, path in enumerate(checkpoints)
                                     if path.parent.name == "double_nstep_1m_g4_seed1" and path.name == "checkpoint_1000000.pt"), 0)
        checkpoint = st.selectbox("Modèle", checkpoints, index=preferred_checkpoint,
                                  format_func=lambda path: str(path.relative_to(root)))
        test_episodes = st.number_input("Épisodes", min_value=1, max_value=20, value=3)
        test_seed = st.number_input("Seed de départ", min_value=0, value=8001)
        if st.button("Lancer l'évaluation", type="primary"):
            base = root / f"dashboard_eval_{checkpoint.parent.name}_{checkpoint.stem}_{test_seed}"
            output = base
            index = 2
            while output.exists():
                output = Path(f"{base}_{index}")
                index += 1
            command = [sys.executable, str(Path(__file__).with_name("evaluate.py")),
                       "--checkpoint", str(checkpoint.resolve()), "--episodes", str(test_episodes),
                       "--seed", str(test_seed), "--out", str(output.resolve())]
            with st.spinner("Évaluation en cours…"):
                process = subprocess.run(command, capture_output=True, text=True, timeout=1200)
            if process.returncode:
                st.error(process.stderr[-2000:])
            else:
                st.session_state.setdefault("selected_runs", [])
                st.session_state["selected_runs"].append(output.name)
                st.rerun()
    else:
        st.caption("Un checkpoint sera proposé ici après un entraînement.")
folders = sorted({path.parent for path in root.glob("**/episodes.jsonl")}) if root.exists() else []
if not folders:
    st.info("Aucun résultat pour l'instant. Lancez evaluate.py pour l'agent aléatoire, puis train.py.")
    st.code("python evaluate.py --episodes 50 --seed 7001 --out runs/random_seed7001\npython train.py --out runs/new_run")
    st.stop()
labels = {str(folder.relative_to(root)): folder for folder in folders}
if "selected_runs" not in st.session_state:
    preferred = [name for name in ("final_1m_g4_seed1", "final_1m_g4_seed2", "random_seed7001") if name in labels]
    st.session_state["selected_runs"] = preferred or list(labels)[:3]
selected = st.multiselect("Expériences à comparer", list(labels), key="selected_runs",
                          format_func=display_name)
if not selected:
    st.stop()
data = {}
for label in selected:
    episodes = read_episodes(labels[label] / "episodes.jsonl")
    if episodes:
        data[label] = (episodes, summarize(episodes))
if not data:
    st.warning("Les fichiers d'épisodes sélectionnés sont vides.")
    st.stop()
summary = pd.DataFrame([{"Expérience": display_name(label), "Épisodes": stats["episodes"],
                         "Reward moyen": stats["reward_mean"], "Score Crafter (%)": stats["score"],
                         "Longueur moyenne": stats["length_mean"]}
                        for label, (_, stats) in data.items()])
st.subheader("Comparaison")
st.dataframe(summary.style.format({"Reward moyen": "{:.2f}", "Score Crafter (%)": "{:.2f}",
                                   "Longueur moyenne": "{:.0f}"}), hide_index=True, width="stretch")
left, right = st.columns(2)
with left:
    st.plotly_chart(px.bar(summary, x="Expérience", y="Reward moyen", color="Expérience",
                           title="Reward épisodique moyen"), width="stretch")
with right:
    st.plotly_chart(px.bar(summary, x="Expérience", y="Score Crafter (%)", color="Expérience",
                           title="Score géométrique des achievements"), width="stretch")
training_figure = Path("figures/training_reward.png")
if training_figure.exists():
    st.subheader("Progression pendant l'entraînement")
    st.image(str(training_figure),
             caption="Reward moyen par tranche de 10 000 pas · moyenne de deux entraînements · zone bleue : dispersion entre les seeds")
skills = []
for label, (_, stats) in data.items():
    skills += [{"Expérience": display_name(label), "Achievement": name.replace("_", " "), "Succès (%)": rate}
               for name, rate in stats["success"].items()]
if skills:
    st.subheader("Compétences débloquées")
    st.plotly_chart(px.bar(pd.DataFrame(skills), x="Succès (%)", y="Achievement", color="Expérience",
                           barmode="group", orientation="h", height=750), width="stretch")
st.caption("Score : moyenne géométrique des 22 taux de succès, exprimés en pourcentage. Les courbes de train décrivent les épisodes observés pendant l'apprentissage.")
