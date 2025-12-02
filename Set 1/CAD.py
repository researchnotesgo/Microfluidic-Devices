
# CAD_outward15_Tfix.py
# ✅ T-junction fixed: single vertical branch from center upwards (no minus branch) → true "T"
# ✅ Circular hollow sweep for Spiral, T, and Y
# ✅ Y angle = 45°
# ✅ Separate polylines in DXF/SVG to avoid accidental connections
# ✅ 0.2 mm outward, wall_ratio = 0.30; bold ~70% annotations
# ✅ GPT-5 SVG primary (10s watchdog), local fallback
# ✅ OpenSCAD→Trimesh union fallback; watertight checks; CSV + prompt logs

import os, math, time, uuid, hashlib, tempfile, subprocess, csv, re, textwrap
import numpy as np
import trimesh, svgwrite, ezdxf
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

OUTDIR = "fdm_outward_fixed300_output"
CSV_PATH = os.path.join(OUTDIR, "device_runs.csv")
PROMPT_LOG = os.path.join(OUTDIR, "prompt_log.txt")
os.makedirs(OUTDIR, exist_ok=True)

OPENSCAD_EXE = r"C:\Program Files\OpenSCAD\openscad.exe"
OUTWARD_THICKEN_MM = 0.2

DEFAULT = {"w":300, "r0":200, "s":2000, "loops":5,
           "h":2000, "v":1500, "angle":45, "wall_ratio":0.30}

API_OK=False
client=None
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    if os.environ.get("OPENAI_API_KEY"):
        API_OK=True
        print("🧠 OpenAI API connected (GPT-5).")
    else:
        print("⚠️ OPENAI_API_KEY not set → using local SVGs.")
except Exception as e:
    print(f"⚠️ OpenAI client unavailable: {e}")

def log_prompt(kind, text):
    with open(PROMPT_LOG, "a", encoding="utf-8") as f:
        f.write(f"\n{'-'*60}\n{time.strftime('%Y-%m-%d %H:%M:%S')} | {kind}\n{text}\n")

def um2mm(x): return np.asarray(x,float)/1000.0
def safe_norm(v): n=np.linalg.norm(v); return v/n if n>1e-12 else v

def md5_of_file(path):
    if not os.path.exists(path): return "N/A"
    h=hashlib.md5()
    with open(path,"rb") as f:
        for c in iter(lambda:f.read(4096), b""): h.update(c)
    return h.hexdigest()

# ---------- circular sweep ----------
def parallel_transport(P):
    P=np.asarray(P,float); N=len(P)
    T,Nn,Bb=np.zeros((N,3)),np.zeros((N,3)),np.zeros((N,3))
    for i in range(N):
        if i==0: T[i]=safe_norm(P[1]-P[0])
        elif i==N-1: T[i]=safe_norm(P[i]-P[i-1])
        else: T[i]=safe_norm(P[i+1]-P[i-1])
    up=np.array([0,0,1.])
    if abs(np.dot(up,T[0]))>0.95: up=np.array([1.,0,0])
    Nn[0]=safe_norm(np.cross(T[0],np.cross(up,T[0]))); Bb[0]=safe_norm(np.cross(T[0],Nn[0]))
    for i in range(1,N):
        v=safe_norm(np.cross(T[i-1],T[i])); s=np.linalg.norm(np.cross(T[i-1],T[i]))
        c=float(np.clip(np.dot(T[i-1],T[i]),-1.,1.))
        if s<1e-6: Nn[i]=Nn[i-1]; Bb[i]=safe_norm(np.cross(T[i],Nn[i])); continue
        θ=math.atan2(s,c); K=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        R=np.eye(3)+math.sin(θ)*K+(1-math.cos(θ))*(K@K)
        Nn[i]=safe_norm(R@Nn[i-1]); Bb[i]=safe_norm(np.cross(T[i],Nn[i]))
    return T,Nn,Bb

def sweep_hollow_tube(points_mm, outer_d_mm, wall_mm, segments=64):
    P=np.asarray(points_mm,float)
    R=max(outer_d_mm/2,1e-6); r=max(R-wall_mm,1e-6)
    if r>=R: r=max(0.9*R,1e-6)
    T,Nn,Bb=parallel_transport(P)
    θ=np.linspace(0,2*np.pi,segments,endpoint=False); ct,st=np.cos(θ),np.sin(θ)
    Vout,Vin=[],[]
    for i in range(len(P)):
        n,b,c=Nn[i],Bb[i],P[i]
        Vout.append(c+(n[None,:]*ct[:,None]+b[None,:]*st[:,None])*R)
        Vin.append( c+(n[None,:]*ct[:,None]+b[None,:]*st[:,None])*r)
    Vout=np.vstack(Vout); Vin=np.vstack(Vin)
    def ring_faces(off,rings,segs,flip=False):
        F=[]
        for i in range(rings-1):
            a0=off+i*segs; a1=off+(i+1)*segs
            for j in range(segs):
                a=a0+j; b=a0+(j+1)%segs; c=a1+j; d=a1+(j+1)%segs
                F += [[a,c,b],[b,c,d]]
        F=np.asarray(F,int); return F[:,::-1] if flip else F
    rings=len(P); segs=segments
    Fout=ring_faces(0,rings,segs,flip=False)
    Fin=ring_faces(len(Vout),rings,segs,flip=True)
    V=np.vstack([Vout,Vin]); F=np.vstack([Fout,Fin])
    m=trimesh.Trimesh(vertices=V,faces=F,process=True)
    m.fix_normals()
    if not m.is_watertight:
        print("⚠️ Sweep not watertight; repairing…")
        m.fill_holes(); m.remove_duplicate_faces(); m.remove_unreferenced_vertices()
    return m

def outward_thicken_params(outer_d_mm, wall_mm, delta_mm):
    return outer_d_mm+2*delta_mm, wall_mm+delta_mm

# ---------- union/export ----------
def export_union(meshes,outpath):
    concat=trimesh.util.concatenate(meshes); concat.fix_normals()
    try:
        if OPENSCAD_EXE and os.path.exists(OPENSCAD_EXE):
            tmp=tempfile.mkdtemp(prefix="scad_union_"); stls=[]
            for i,m in enumerate(meshes):
                p=os.path.join(tmp,f"p{i}.stl"); m.export(p); stls.append(p.replace('\\','/'))
            scad=os.path.join(tmp,"union.scad")
            with open(scad,"w") as f:
                f.write("union(){\n"); [f.write(f'import(\"{s}\");\n') for s in stls]; f.write("}\n")
            subprocess.run([OPENSCAD_EXE,"-o",outpath,scad],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            print(f"✅ OpenSCAD union → {outpath}"); return
    except Exception as e:
        print(f"⚠️ OpenSCAD union failed: {e}")
    concat.export(outpath); print(f"✅ Trimesh union → {outpath}")

# ---------- annotations & drawing ----------
def annotation_text(kind, p):
    extra = ""
    if kind=="Spiral":
        extra=f" | loops={p.loops} r0={p.r0}µm s={p.s}µm"
    elif kind=="T-junction":
        extra=f" | h={p.h}µm v={p.v}µm"
    elif kind=="Y-junction":
        extra=f" | h={p.h}µm v={p.v}µm θ={p.angle}°"
    return f"{kind} | w={p.w}µm | outer+0.2mm | wall_ratio={p.wall_ratio}{extra}"

def export_dxf_multi(kind, paths_um, p, base):
    doc=ezdxf.new(setup=True); msp=doc.modelspace()
    outer_mm=p.w/1000.0; wall_mm=(p.w*p.wall_ratio)/1000.0
    outer2_mm,_=outward_thicken_params(outer_mm,wall_mm,OUTWARD_THICKEN_MM)
    outer2_um=outer2_mm*1000.0
    all_pts=[]

    for path in paths_um:
        c2=np.asarray(path,float)[:,:2]
        msp.add_lwpolyline(c2, dxfattribs={"const_width": outer2_um/2.0, "layer":"0"})
        all_pts.append(c2)

    all_pts=np.vstack(all_pts)
    xmin,xmax=float(np.min(all_pts[:,0])),float(np.max(all_pts[:,0]))
    ymin,ymax=float(np.min(all_pts[:,1])),float(np.max(all_pts[:,1]))
    W,H=xmax-xmin,ymax-ymin
    cx, cy = (xmin+xmax)/2.0, ymax + 0.25*H
    text_height = max(35.0, 0.056*H)
    text_str = annotation_text(kind,p)
    t1 = msp.add_text(text_str, dxfattribs={"height":text_height,"layer":"0"})
    t2 = msp.add_text(text_str, dxfattribs={"height":text_height,"layer":"0"})
    try:
        t1.set_placement((cx,cy)); t2.set_placement((cx+0.02*text_height, cy))
    except Exception:
        t1.dxf.insert=(cx,cy); t2.dxf.insert=(cx+0.02*text_height, cy)
    out=f"{base}.dxf"; doc.saveas(out); print(f"📐 DXF (multi) → {out}"); return out

def export_svg_local_multi(kind, paths_um, p, base):
    all_pts=np.vstack([np.asarray(path,float)[:,:2] for path in paths_um])
    xmin,xmax=float(np.min(all_pts[:,0])),float(np.max(all_pts[:,0]))
    ymin,ymax=float(np.min(all_pts[:,1])),float(np.max(all_pts[:,1]))
    W,H=xmax-xmin,ymax-ymin; margin=0.45*max(W,H)
    vb_minx,vb_miny=xmin-margin,ymin-margin
    vb_width,vb_height=W+2*margin,H+2*margin

    outer_mm=p.w/1000.0; wall_mm=(p.w*p.wall_ratio)/1000.0
    outer2_mm,_=outward_thicken_params(outer_mm,wall_mm,OUTWARD_THICKEN_MM)
    stroke_width_um=(outer2_mm*1000.0)/3.0

    dwg=svgwrite.Drawing(f"{base}_model.svg", size=("120mm","120mm"))
    dwg.viewbox(vb_minx,vb_miny,vb_width,vb_height)
    for path in paths_um:
        c2=np.asarray(path,float)[:,:2]
        dwg.add(dwg.polyline(c2.tolist(), stroke="black", fill="none", stroke_width=stroke_width_um))
    cx, cy = (xmin+xmax)/2.0, ymax + 0.25*H
    fs = max(7.0, 0.042*max(W,H))
    dwg.add(dwg.text(annotation_text(kind,p), insert=(cx, cy),
                     text_anchor="middle", font_size=f"{fs}", font_family="Arial",
                     font_weight="bold", fill="black"))
    dwg.save(); print(f"🖼️ SVG (multi) → {base}_model.svg"); return f"{base}_model.svg"

def export_svg_gpt_multi(kind, paths_um, p, base):
    poly_blocks=[np.round(np.asarray(path,float)[:,:2],3).tolist() for path in paths_um]
    outer_mm=p.w/1000.0; wall_mm=(p.w*p.wall_ratio)/1000.0
    outer2_mm,_=outward_thicken_params(outer_mm,wall_mm,OUTWARD_THICKEN_MM)
    stroke_width_um=(outer2_mm*1000.0)/3.0
    prompt=textwrap.dedent(f"""
        Generate an SVG (no code fences) for a {kind} microfluidic channel.
        Draw multiple separate polylines; do not connect them.
        Polyline sets (µm): {poly_blocks}
        Requirements:
        - viewBox fits geometry with ~45% margin.
        - Stroke black, stroke-width ≈ {stroke_width_um:.3f}.
        - Bold annotation (~4.2% of geometry height) near top-center: "{annotation_text(kind,p)}".
        - Return valid SVG XML only.
    """).strip()
    log_prompt(kind,prompt)
    if not API_OK:
        print("⚠️ API unavailable → local SVG."); return export_svg_local_multi(kind,paths_um,p,base)
    def call_api():
        resp=client.chat.completions.create(
            model="gpt-5",
            messages=[{"role":"system","content":"You are an expert CAD assistant. Return SVG XML only."},
                      {"role":"user","content":prompt}],
            max_completion_tokens=1600)
        return resp.choices[0].message.content.strip()
    try:
        with ThreadPoolExecutor() as ex:
            svg_code=ex.submit(call_api).result(timeout=10)
        if svg_code.startswith("```"):
            m=re.search(r"```(?:xml|svg)?\s*(.*?)```",svg_code,re.DOTALL|re.IGNORECASE)
            if m: svg_code=m.group(1).strip()
        out=f"{base}_gpt.svg"
        with open(out,"w",encoding="utf-8") as f: f.write(svg_code)
        print(f"🧠 GPT SVG → {out}"); return out
    except: return export_svg_local_multi(kind,paths_um,p,base)

# ---------- builders ----------
def build_spiral(p):
    print("🔄 Spiral...")
    th=np.linspace(0,2*np.pi*p.loops,1200)
    r=p.r0+p.s*th/(2*np.pi)
    x,y=r*np.cos(th),r*np.sin(th)
    path_um=np.column_stack([x,y,np.zeros_like(x)])
    path_mm=um2mm(path_um)
    outer=um2mm(p.w); wall=um2mm(p.w*p.wall_ratio)
    outer2,wall2=outward_thicken_params(outer,wall,OUTWARD_THICKEN_MM)
    tube=sweep_hollow_tube(path_mm, outer2, wall2, 64)
    base=os.path.join(OUTDIR,"spiral_outward15")
    stl=f"{base}.stl"; export_union([tube], stl)
    dxf=export_dxf_multi("Spiral", [path_um], p, base)
    svg=export_svg_gpt_multi("Spiral", [path_um], p, base)
    return stl,dxf,svg

def build_tjunction(p):
    print("🔄 T-junction (fixed: upward branch only)...")
    # Main along X through the origin
    main_um=np.array([[-p.h/2,0,0],[p.h/2,0,0]])
    # Branch ONLY upward from center (true T)
    branch_um=np.array([[0,0,0],[0,p.v,0]])
    outer=um2mm(p.w); wall=um2mm(p.w*p.wall_ratio)
    outer2,wall2=outward_thicken_params(outer,wall,OUTWARD_THICKEN_MM)
    tube1=sweep_hollow_tube(um2mm(main_um), outer2, wall2, 64)
    tube2=sweep_hollow_tube(um2mm(branch_um), outer2, wall2, 64)
    base=os.path.join(OUTDIR,"tjunction_outward15")
    stl=f"{base}.stl"; export_union([tube1,tube2], stl)
    dxf=export_dxf_multi("T-junction", [main_um, branch_um], p, base)
    svg=export_svg_gpt_multi("T-junction", [main_um, branch_um], p, base)
    return stl,dxf,svg

def build_yjunction(p):
    print("🔄 Y-junction (±angle, 45°)...")
    ang=math.radians(p.angle)
    main_um=np.array([[-p.h/2,0,0],[0,0,0]])
    b1_um=np.array([[0,0,0],[p.v*math.cos(ang), p.v*math.sin(ang),0]])
    b2_um=np.array([[0,0,0],[p.v*math.cos(ang),-p.v*math.sin(ang),0]])
    outer=um2mm(p.w); wall=um2mm(p.w*p.wall_ratio)
    outer2,wall2=outward_thicken_params(outer,wall,OUTWARD_THICKEN_MM)
    tube_main=sweep_hollow_tube(um2mm(main_um), outer2, wall2, 64)
    tube_b1  =sweep_hollow_tube(um2mm(b1_um),   outer2, wall2, 64)
    tube_b2  =sweep_hollow_tube(um2mm(b2_um),   outer2, wall2, 64)
    base=os.path.join(OUTDIR,"yjunction_outward15")
    stl=f"{base}.stl"; export_union([tube_main,tube_b1,tube_b2], stl)
    dxf=export_dxf_multi("Y-junction", [main_um, b1_um, b2_um], p, base)
    svg=export_svg_gpt_multi("Y-junction", [main_um, b1_um, b2_um], p, base)
    return stl,dxf,svg

def main():
    p=SimpleNamespace(**DEFAULT)
    print("=== Trio (T fixed; circular STL; angle=45°) ===")
    session=str(uuid.uuid4())[:8]
    rows=[]
    for fn,label in [(build_spiral,"Spiral"),(build_tjunction,"T-junction"),(build_yjunction,"Y-junction")]:
        t0=time.time(); stl,dxf,svg=fn(p); dt=time.time()-t0
        rows.append({"session":session,"device":label,"t_s":f"{dt:.3f}","stl":stl,"dxf":dxf,"svg":svg,"stl_md5":md5_of_file(stl)})
    with open(CSV_PATH,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys())
        if f.tell()==0: w.writeheader()
        for r in rows: w.writerow(r)
    print(f"✅ Done → {OUTDIR}\n📊 CSV → {CSV_PATH}\n🧠 Log → {PROMPT_LOG}")

if __name__=="__main__": main()
