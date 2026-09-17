"""Cutilytics AI — non-clinical skincare-exploration MVP."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Cutilytics AI", page_icon="✦", layout="wide")
st.markdown("""<style>
.stApp{background:#fbfaff;color:#222046}.block-container{max-width:1460px;padding:1rem 2.5rem 2rem}.top{background:linear-gradient(100deg,#eadcff 0%,#fbf8ff 47%,#e3f1d8 100%);border:1px solid #d9d1e8;border-radius:0 0 24px 24px;padding:1.05rem 2.2rem;display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem}.brand{font-size:2.15rem;font-weight:800;letter-spacing:-.04em}.brand span{color:#74a163}.tag{font-size:1rem;color:#383052;margin-top:.1rem}.nav{font-size:.76rem;font-weight:700;border:1px solid #8aa381;border-radius:22px;padding:.6rem 1rem;background:#fffdfdcc}.leaf{color:#78a569;font-size:1.8rem}.panel{border:1px solid #ded9eb;border-radius:17px;background:rgba(255,255,255,.7);padding:1.25rem;margin:.4rem 0}.intro{padding:1.15rem 1.5rem}.intro h2{margin:0;color:#25214c}.muted{color:#625d79}.steps{display:flex;justify-content:space-evenly;align-items:center;padding-top:.8rem}.step{text-align:center;font-weight:700}.round{height:67px;width:67px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:1.75rem;margin:0 auto .45rem;background:#eee5ff}.round.green{background:#e7f4db}.arrow{color:#aa9cbf;font-size:1.8rem}.section-title{font-weight:800;font-size:1.16rem}.small{font-size:.87rem;color:#68627b}.profile{background:linear-gradient(105deg,#f1e7ff,#eff8e8);border-radius:12px;padding:1.25rem;min-height:142px}.hint{background:#f2edff;border-radius:12px;padding:1.25rem;min-height:142px}.status{margin-top:.7rem;background:#ede8f7;border-radius:20px;padding:.55rem .8rem;text-align:center;color:#8062a7;font-weight:700}.recommend{background:#f2f8f0;border-radius:11px;padding:.8rem 1rem;color:#526a54;font-size:.86rem}.disclaimer{border:1px solid #ded9eb;border-radius:12px;padding:.8rem 1rem;color:#645d78;font-size:.78rem}.blueprint{background:linear-gradient(100deg,#eee0ff,#eff6df);padding:1rem 1.3rem;border-radius:14px}.coming{display:inline-block;color:#4f8154;background:#e4f1df;border-radius:12px;padding:.25rem .6rem;font-size:.75rem;font-weight:700;margin-left:.6rem}div[data-testid='stFileUploader']{background:#faf8ff;border:2px dashed #d7cde5;border-radius:12px;padding:.9rem}.stButton>button{width:100%;border:0;border-radius:24px;background:linear-gradient(90deg,#ddc5fb,#d7ecd0);color:#493d67;font-weight:800;padding:.65rem}.indicator{padding:.65rem 0;border-bottom:1px solid #e9e5f0}.indlabel{font-weight:700;font-size:.9rem}</style>""",unsafe_allow_html=True)

def cues(image):
    px=np.asarray(image.convert('RGB').resize((224,224)),dtype=np.float32)
    bright,contrast=px.mean(),px.std();warm=(px[...,0].mean()-px[...,1].mean())/255
    return [min(100,round(abs(warm)*420)),min(100,round(max(0,65-contrast)*1.5)),min(100,round(max(0,50-contrast)*1.7))]

def predict_pose(image):
    return 'Photo ready for analysis', None

st.markdown("<div class='top'><div><span class='leaf'>❋</span> <span class='brand'>Cutilytics <span>AI</span></span><div class='tag'>Nurture your natural glow</div></div><div class='nav'>AI &nbsp;•&nbsp; PERSONALIZED SKINCARE &nbsp;•&nbsp; MVP</div><span class='leaf'>❧</span></div>",unsafe_allow_html=True)

intro,flow=st.columns([.46,.54])
with intro:
    st.markdown("<div class='panel intro'><h2>Meet Your Skin Profile</h2><p class='muted'>Cutilytics AI begins with one focused feature:<br><b>AI-powered skin image analysis.</b><br>Upload a facial image and explore a preliminary profile of visible image characteristics. This MVP validates the image-analysis step before expanding the platform.</p></div>",unsafe_allow_html=True)
with flow:
    st.markdown("<div class='panel'><div class='steps'><div class='step'><div class='round'>▣</div>Image Analysis</div><div class='arrow'>→</div><div class='step'><div class='round green'>▥</div>Skin Profile</div><div class='arrow'>→</div><div class='step'><div class='round'>◒</div>Ingredient Blueprint<br><span class='small'>(Coming Soon)</span></div></div></div>",unsafe_allow_html=True)

left,right=st.columns([.41,.59],gap='medium')
with left:
    st.markdown("<div class='panel'><div class='section-title'>▣ &nbsp; Upload Your Skin Image</div><div class='small'>For the best preliminary analysis, use a clear, well-lit image with minimal makeup and no filters.</div></div>",unsafe_allow_html=True)
    upload=st.file_uploader('Drag & drop your image here',type=['jpg','jpeg','png'],label_visibility='collapsed')
    camera=st.camera_input('Or take a live photo')
    image_source=camera if camera is not None else upload
    st.markdown("<div class='recommend'><b>◉ &nbsp; Recommended</b><br>• Clear image &nbsp;&nbsp;&nbsp; • No beauty filters<br>• Face centered &nbsp;&nbsp;&nbsp; • Minimal makeup</div>",unsafe_allow_html=True)
    if image_source is not None:
        image=Image.open(image_source).convert('RGB');st.image(image,caption='Your camera photo' if camera is not None else 'Your uploaded image',use_container_width=True)
    if st.button('✧ &nbsp; Analyze My Skin',disabled=image_source is None):
        st.session_state['analysed']=True
    st.markdown("<div class='small' style='margin-top:.75rem'><b>◎ &nbsp; Analysis Status</b><br>"+('Analysis complete.' if st.session_state.get('analysed') else 'Waiting for image…')+"</div>",unsafe_allow_html=True)
with right:
    st.markdown("<div class='panel'><div class='section-title'>❧ &nbsp; Your Skin Profile</div><div class='small'>Your preliminary results will appear below.</div><hr style='border:.5px solid #ebe6f0'>",unsafe_allow_html=True)
    if image_source is not None and st.session_state.get('analysed'):
        pose,confidence=predict_pose(image);values=cues(image)
        a,b,c=st.columns([1.15,.72,.95])
        with a: st.markdown(f"<div class='profile'><b>◌ &nbsp; Overall Skin Profile</b><div class='status'>{pose}"+(f" · {confidence:.0%}" if confidence is not None else '')+"</div><div class='small' style='margin-top:.55rem'>Preliminary capture guidance only.</div></div>",unsafe_allow_html=True)
        with b: st.image(image,use_container_width=True)
        with c: st.markdown("<div class='hint'>✧<br><br>Image received. Capture angle and photo cues are ready to support the MVP workflow.</div>",unsafe_allow_html=True)
        st.markdown("<div class='section-title' style='margin-top:1rem'>▥ &nbsp; Visible Image Indicators</div>",unsafe_allow_html=True)
        for label,value,color in zip(['Tone appearance','Colour warmth','Texture visibility'],values,['#b58dea','#9ecb86','#b58dea']):
            st.markdown(f"<div class='indicator'><span class='indlabel'>{label}</span>",unsafe_allow_html=True);st.progress(value,text=f'{value}/100');st.markdown('</div>',unsafe_allow_html=True)
    else:
        st.markdown("<div class='profile'><b>◌ &nbsp; Overall Skin Profile</b><div class='status'>Awaiting analysis…</div></div>",unsafe_allow_html=True)
        st.markdown("<div class='hint' style='margin-top:.7rem'>✧<br><br>Upload an image and click <b>Analyze My Skin</b> to begin.</div></div>",unsafe_allow_html=True)

st.markdown("<div class='blueprint'><b>❧ &nbsp; Personalized Ingredient Blueprint</b><span class='coming'>Coming Soon</span><br><span class='small'>Once the profile engine is validated, the platform can progress toward personalized ingredient education.</span></div>",unsafe_allow_html=True)
st.markdown("<div class='disclaimer' style='margin-top:1rem'><b>ⓘ &nbsp; Prototype Notice:</b> Cutilytics AI is an MVP prototype. The current analysis provides preliminary, non-clinical image guidance and is not a medical diagnosis or substitute for professional dermatological advice.</div>",unsafe_allow_html=True)
st.caption('© 2026 Cutilytics AI')
