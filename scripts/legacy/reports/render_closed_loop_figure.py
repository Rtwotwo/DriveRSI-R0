#!/usr/bin/env python3
"""Render the versioned DriveRSI closed-loop FigureSpec as editable SVG."""
import argparse
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

NS="http://www.w3.org/2000/svg"
ET.register_namespace("",NS)
PALETTE={"blue":("#2F66C6","#EEF4FF"),"orange":("#BB721C","#FFF6E8"),
         "purple":("#8054C8","#F4F0FC"),"teal":("#148672","#ECF8F3"),
         "slate":("#566D84","#F0F4F8")}
def render(spec):
    w,h=spec["canvas"]["width"],spec["canvas"]["height"]
    svg=ET.Element("{%s}svg"%NS,{"viewBox":f"0 0 {w} {h}","width":str(w),"height":str(h),
      "font-family":"'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif",
      "role":"img","aria-labelledby":"figure-title figure-description"})
    def el(tag,attrs=None,value=None,parent=svg):
        node=ET.SubElement(parent,"{%s}%s"%(NS,tag),{k:str(v) for k,v in (attrs or {}).items()})
        if value is not None:node.text=value
        return node
    def tx(x,y,value,size=18,color="#34475D",weight=400,anchor="start",parent=svg):
        return el("text",{"x":x,"y":y,"font-size":size,"fill":color,"font-weight":weight,"text-anchor":anchor},value,parent)
    def rect(x,y,ww,hh,fill,stroke="none",radius=12,parent=svg):
        return el("rect",{"x":x,"y":y,"width":ww,"height":hh,"rx":radius,"fill":fill,"stroke":stroke},parent=parent)
    el("title",{"id":"figure-title"},spec["title"])
    el("desc",{"id":"figure-description"},"环形主线加直接 RL 与免渲染旁路；候选按任务分流，模型和改进器独立晋级。此图不代表全部模块已完成验收。")
    el("metadata",value=json.dumps({k:spec[k] for k in ["date","commit","evidence_scope"]},ensure_ascii=False))
    defs=el("defs")
    for key,(c,_) in PALETTE.items():
        marker=el("marker",{"id":"arrow-"+key,"viewBox":"0 0 10 10","refX":9,"refY":5,
          "markerWidth":7,"markerHeight":7,"orient":"auto"},parent=defs)
        el("path",{"d":"M 1 1 L 9 5 L 1 9 Z","fill":c},parent=marker)
    rect(0,0,w,h,"#F8FAFD",radius=0)
    rect(0,0,w,174,"#13263C",radius=0)
    rect(80,38,5,83,"#67D2B5",radius=2)
    tx(104,60,"DRIVERSI  /  END-TO-END IMPROVEMENT LOOP",16,"#95B6D7",600)
    tx(104,112,spec["title"],44,"#FFFFFF",700)
    tx(104,146,spec["subtitle"],17,"#C8D7E8")
    tx(1920,54,spec["date"]+"  ·  SERVER 32341",16,"#BED1E3",500,"end")
    tx(1920,81,spec["commit"],14,"#A1BAD1",400,"end")
    for x,c,t in [(1480,"#78A5ED","实线：数据 / 执行"),(1740,"#67D2B5","虚线：版本反馈")]:
        el("circle",{"cx":x,"cy":124,"r":6,"fill":c});tx(x+14,130,t,16,"#F0F5FA")
    # Main process and the single controlled feedback path.
    nodes={n["id"]:n for n in spec["nodes"]}
    assert len(nodes)==len(spec["nodes"])
    for e in spec["edges"]:
        assert e["from"] in nodes and e["to"] in nodes
        p=e["points"];assert len(p)>1
        assert all(0<=x<=w and 0<=y<=h for x,y in p)
        attrs={"id":e["id"],"d":"M "+" L ".join(f"{x} {y}" for x,y in p),
          "fill":"none","stroke":PALETTE[e["color"]][0],"stroke-width":2.6,
          "stroke-linejoin":"round","marker-end":f"url(#arrow-{e['color']})"}
        if e["dashed"]:attrs["stroke-dasharray"]="9 7"
        el("path",attrs)
    for n in spec["nodes"]:
        x=n["x"]-n["width"]/2;y=n["y"]-n["height"]/2
        assert 0<=x and 0<=y and x+n["width"]<=w and y+n["height"]<=h
        c,bg=PALETTE[n["tone"]];g=el("g",{"id":n["id"]})
        el("title",value=n["label"],parent=g)
        rect(x,y,n["width"],n["height"],"#FFFFFF","#D4DFEB",14,g)
        rect(x,y,5,n["height"],c,radius=2,parent=g)
        el("circle",{"cx":x+33,"cy":y+33,"r":18,"fill":bg},parent=g)
        tx(x+33,y+39,n["number"],17,c,700,"middle",g)
        tx(x+62,y+41,n["label"],24,c,700,parent=g)
        tx(x+20,y+74,n["subtitle"],16,"#6C7C8E",parent=g)
        for i,line in enumerate(n["body"]):tx(x+20,y+108+i*26,line,18,"#35485C",parent=g)
        tx(x+20,y+n["height"]-16,n["foot"],12,"#8391A3",parent=g)
    for b in spec["supports"]:
        x,y=b["x"],b["y"]
        rect(x,y,b["width"],b["height"],"#EEF3F8","#D4DFEB")
        tx(x+22,y+35,b["title"],23,"#36546D",700)
        tx(x+22,y+63,b["subtitle"],16,"#6A8094")
        for i,line in enumerate(b["lines"]):
            el("circle",{"cx":x+26,"cy":y+92+i*29,"r":3,"fill":"#7994AC"})
            tx(x+39,y+98+i*29,line,17,"#445E75")
    # Conceptual interaction scene: pure vectors, never a claimed rendered frame.
    center=el("g",{"id":"causal-data-paths","transform":"translate(560 500)"})
    def sh(tag,attrs):return el(tag,attrs,parent=center)
    def ct(x,y,value,size=16,color="#40566C",weight=400,anchor="start"):
        return tx(x,y,value,size,color,weight,anchor,center)
    def bx(x,y,ww,hh,fill,stroke="none",rr=9):return rect(x,y,ww,hh,fill,stroke,rr,center)
    def line(d,color,width=3,dash="",marker=""):
        a={"d":d,"fill":"none","stroke":color,"stroke-width":width,"stroke-linecap":"round","stroke-linejoin":"round"}
        if dash:a["stroke-dasharray"]=dash
        if marker:a["marker-end"]=f"url(#arrow-{marker})"
        return sh("path",a)
    def car(x,y,color,angle=0):
        g=el("g",{"transform":f"translate({x} {y}) rotate({angle})"},parent=center)
        rect(-15,-27,30,54,color,"#FFFFFF",7,g)
        el("path",{"d":"M -10 -16 H 10 L 8 -7 H -8 Z","fill":"#D5E8F4"},parent=g)
        rect(-9,13,18,7,"#C2DCEA",radius=2,parent=g)
        el("path",{"d":"M -4 -22 L 0 -25 L 4 -22","fill":"none","stroke":"#FFFFFF","stroke-width":1.5},parent=g)
    bx(0,0,900,700,"#FFFFFF","#D4DFEB",16)
    ct(25,35,spec["center"]["title"],25,"#214762",700)
    ct(25,61,spec["center"]["subtitle"],17,"#6B7F91")
    bx(24,90,852,220,"#E7EFE9","#D8E2DC",12)
    for x,y,ww,hh in [(43,107,133,47),(553,109,90,43),(52,260,190,35)]:
        bx(x,y,ww,hh,"#CAD9E2","#C0D0DB",5)
        for k in range(int(ww/24)-1):bx(x+10+k*24,y+10,11,9,"#EAF0F5",rr=1)
    bx(393,90,140,220,"#697B8D",rr=0)
    bx(24,164,646,90,"#697B8D",rr=0)
    line("M 463 90 V 164 M 463 254 V 310","#F5DB91",2,"9 8")
    line("M 24 208 H 393 M 533 208 H 670","#F5DB91",2,"9 8")
    for xx in range(402,527,20):bx(xx,149,10,15,"#F5F8FB",rr=1);bx(xx,263,10,15,"#F5F8FB",rr=1)
    sh("path",{"d":"M 493 274 L 408 108 L 579 108 Z","fill":"#BAA0E3","opacity":.25,
              "stroke":"#9871CC","stroke-width":1.5,"stroke-dasharray":"4 4"})
    line("M 204 208 H 623","#D78B24",4,marker="orange")
    line("M 493 280 C 494 247 487 218 460 208","#D95B63",3,"6 5")
    line("M 493 280 C 493 251 478 238 446 238 H 291","#159782",3,marker="teal")
    sh("circle",{"cx":463,"cy":208,"r":25,"fill":"#F5AAAA","fill-opacity":.2,"stroke":"#F3BCBC","stroke-dasharray":"4 4"})
    line("M 455 200 L 471 216 M 471 200 L 455 216","#FFDADB",2)
    car(493,280,"#2F6ECD");car(193,208,"#DD962F",90);car(431,119,"#99ACBA",180);car(607,181,"#99ACBA",-90)
    ct(81,191,"历史对手 / 抢道",15,"#875719",600)
    ct(539,295,"自车",15,"#245B96",600)
    ct(536,129,"视野",14,"#71529F",600)
    ct(258,287,"候选纠正",14,"#177C68",600)
    ct(545,237,"风险",14,"#8F3E49",600)
    bx(688,114,166,126,"#F5F2FC","#D6C9E9",9)
    ct(700,136,"按世界状态生成 RGB",13,"#7556A1",600)
    bx(700,146,142,77,"#A5CBE2",rr=4);bx(700,182,142,41,"#556575",rr=0)
    sh("path",{"d":"M 711 223 L 759 182 H 788 L 833 223 Z","fill":"#748394"})
    line("M 773 188 L 771 199 M 770 208 L 769 220","#E9DFB5",2)
    bx(793,177,28,23,"#DEA04C",rr=3);bx(797,181,20,6,"#D1E6EF",rr=1)
    line("M 579 123 H 672 V 181 H 696","#8B65B7",2,marker="purple")
    ct(688,267,"示意图，非实测 GS",13,"#8B7A9C")
    ct(25,333,"视觉控制：当前世界 → 渲染 → 决策 → 推进；数据合成：分支记录 → 离线渲染",15,"#4A677F")
    for i,p in enumerate(spec["center"]["paths"]):
        yy=352+i*158
        tone="teal" if i==0 else "purple";c,bg=PALETTE[tone]
        bx(24,yy,852,142,bg,rr=11)
        ct(43,yy+28,p["title"],20,c,700)
        ct(43,yy+51,p["subtitle"],14,"#667B8E")
        for j,label in enumerate(p["steps"]):
            xx=43+j*275
            bx(xx,yy+65,250,33,"#FFFFFF",rr=6)
            ct(xx+125,yy+87,label,16,c,600,"middle")
            if j<2:line(f"M {xx+255} {yy+81} H {xx+268}",c,1.8,marker=tone)
        ct(43,yy+125,p["note"],14,"#61768B")
    ct(24,683,"结构化 / 视觉策略与 RGB 录制分别配置；图像质量和教师资格单独验收。",14,"#7D8B9D")
    # Edge labels stay outside cards and illustrations.
    for e in spec["edges"]:
        if not e["label"]:continue
        width=sum(15 if ord(c)>0x2e80 else 8 for c in e["label"])+18
        if e.get("rotate"):
            lg=el("g",{"transform":f"translate({e['lx']} {e['ly']}) rotate({e['rotate']})"})
            rect(-width/2,-18,width,25,"#F8FAFD",radius=5,parent=lg)
            tx(0,0,e["label"],15,PALETTE[e["color"]][0],500,"middle",lg)
        else:
            rect(e["lx"]-width/2,e["ly"]-18,width,25,"#F8FAFD",radius=5)
            tx(e["lx"],e["ly"],e["label"],15,PALETTE[e["color"]][0],500,"middle")
    rect(80,1540,1860,125,"#EBF1F7",radius=12)
    tx(102,1568,"全流程边界与独立评测",19,"#36546D",700)
    for i,line_text in enumerate(spec["guardrails"]):
        tx(102,1595+i*26,line_text,16,"#587087")
    tx(80,1693,"环形逻辑图；各模块的实现、质量、性能与学习能力分别验收  ·  中央道路与相机为机制示意",14,"#7B8DA0")
    tx(1940,1693,"DriveRSI  ·  assets/figures",14,"#7B8DA0",400,"end")
    ET.indent(svg,space="  ")
    return ET.tostring(svg,encoding="unicode")

def detail_svg(svg_text):
    source=ET.fromstring(svg_text)
    out=ET.Element("{%s}svg"%NS,{"viewBox":"0 0 948 748","width":"100%","height":"auto",
      "style":"display:block;max-width:1200px;height:auto",
      "font-family":source.attrib["font-family"],"role":"img","aria-label":"同一世界的两种数据增强路径"})
    out.append(copy.deepcopy(source.find("{%s}defs"%NS)))
    ET.SubElement(out,"{%s}rect"%NS,{"width":"948","height":"748","fill":"#F8FAFD"})
    scene=copy.deepcopy(source.find(".//*[@id='causal-data-paths']"));scene.set("transform","translate(24 24)");out.append(scene)
    ET.indent(out,space="  ")
    return ET.tostring(out,encoding="unicode")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("spec",type=Path);p.add_argument("--output",required=True,type=Path)
    args=p.parse_args();spec=json.loads(args.spec.read_text(encoding="utf-8"))
    content=render(spec);ET.fromstring(content)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(content,encoding="utf-8")
    args.output.with_name("drive-rsi-data-paths.svg").write_text(detail_svg(content),encoding="utf-8")
    print(json.dumps({"svg":str(args.output),"stages":len(spec["nodes"]),"bytes":len(content.encode())}))
if __name__=="__main__":main()
