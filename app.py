"""Cutilytics AI: non-clinical skincare exploration MVP."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Cutilytics AI", page_icon="✦", layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(circle at 3% 0,#f4eaff,transparent 30%),radial-gradient(circle at 95% 0,#eef6d9,transparent 30%),#fcfbff;color:#38264e}.hero{padding:2rem 0}.eyebrow{color:#76618d;font-size:.75rem;font-weight:700;letter-spacing:.15em}h1{font-size:4rem;line-height:.95;margin:.35rem 0;color:#392650}.tag{font-size:1.25rem;color:#687243;font-weight:600}.card{background:rgba(255,255,255,.78);border:1px solid #e9dff0;border-radius:22px;padding:1.2rem;margin:.5rem 0;box-shadow:0 12px 30px rgba(72,50,93,.06)}.big{font-size:1.4rem;font-weight:700;color:#50386a}.note{background:#fff8e7;border-left:4px solid #b8a052;border-radius:10px;padding:.85rem 1rem;color:#61572d}.pill{display:inline-block;background:#f0e8f8;color:#654c7c;border-radius:999px;padding:.32rem .65rem;margin:.12rem;font-size:.82rem}</style>""",unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_model():
    path=Path("models/cutilitytics_efficientnetb0.keras"); labels=Path("models/class_names.json")
    if not path.exists() or not labels.exists(): return None,[]
    import tensorflow as tf
    return tf.keras.models.load_model(path),json.loads(labels.read_text())

def photo_quality(image):
    px=np.asarray(image.convert("RGB").resize((224,224)),dtype=np.float32)
    bright,contrast=float(px.mean()),float(px.std())
    lighting="well balanced" if 75<=bright<=195 else ("dark" if bright<75 else "very bright")
    return lighting,"clear" if contrast>=35 else "low contrast"

st.markdown("<div class='hero'><div class='eyebrow'>CUTILITYTICS AI · PROTOTYPE</div><h1>Care, made<br>personal.</h1><div class='tag'>Nurture your natural glow</div><p>A gentle visual skincare-exploration experience, built for education—not diagnosis.</p></div>",unsafe_allow_html=True)
left,right=st.columns([1.05,.95],gap="large")
with left:
    st.markdown("### Your skin snapshot")
    upload=st.file_uploader("Upload a clear, front-facing photo",type=["jpg","jpeg","png"],label_visibility="collapsed")
    st.caption("Use even lighting and no filters. Your image is not saved by this app.")
    if upload: image=Image.open(upload);st.image(image,use_container_width=True)
with right:
    st.markdown("### Capture guidance")
    net,classes=load_model()
    if upload and net:
        x=np.expand_dims(np.asarray(image.convert("RGB").resize((224,224)),dtype=np.float32),0)
        scores=net.predict(x,verbose=0)[0]; idx=int(np.argmax(scores))
        st.markdown(f"<div class='card'><div class='big'>Likely {classes[idx].replace('_',' ').title()}</div><p>Capture-angle guidance confidence: {scores[idx]:.0%}.</p></div>",unsafe_allow_html=True)
    elif upload:
        lighting,detail=photo_quality(image)
        st.markdown(f"<div class='card'><div class='big'>{detail.title()} image</div><p>Lighting appears {lighting}. The capture-angle model is preparing.</p></div>",unsafe_allow_html=True)
    else: st.markdown("<div class='card'><div class='big'>Awaiting your photo</div><p>Upload an image to begin the prototype flow.</p></div>",unsafe_allow_html=True)
st.markdown("### Your skincare snapshot")
a,b,c=st.columns(3)
with a: st.markdown("<div class='card'><div class='big'>Photo quality</div><span class='pill'>Lighting</span><span class='pill'>Focus</span><span class='pill'>Angle</span></div>",unsafe_allow_html=True)
with b: st.markdown("<div class='card'><div class='big'>Routine context</div><p>Skin goals, sensitivity and product use matter beyond a photograph.</p></div>",unsafe_allow_html=True)
with c: st.markdown("<div class='card'><div class='big'>Ingredient blueprint</div><p>Personalized education and routine support.</p><span class='pill'>COMING SOON</span></div>",unsafe_allow_html=True)
st.markdown("<div class='note'><b>Prototype notice.</b> Cutilytics AI provides preliminary, non-clinical visual guidance only. It does not diagnose, treat, or rule out any condition. Seek a qualified healthcare professional for concerns.</div>",unsafe_allow_html=True)
st.caption("© Cutilytics AI · Nurture your natural glow")