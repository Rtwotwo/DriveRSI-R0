#!/usr/bin/env python3
"""Render the maintained DriveRSI FigureSpec into a standalone, editable SVG."""
import argparse
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

NS="http://www.w3.org/2000/svg"
ET.register_namespace("",NS)
COLORS={
 "blue":("#2563EB","#EEF4FF"),"purple":("#7953CF","#F5F1FF"),
 "teal":("#138775","#EDF8F5"),"amber":("#B66C13","#FFFAEF"),
 "slate":("#53647B","#F3F5F8")}
STATUS={"implemented":("#138775","已实现接口"),
        "partial":("#C58616","部分接通 / 待验收"),
        "planned":("#8793A5","规划扩展")}
def build(spec):
    w,h=spec["canvas"]["width"],spec["canvas"]["height"]
    ids=[n["id"] for n in spec["nodes"]]
    assert len(ids)==len(set(ids))
    svg=ET.Element("{%s}svg"%NS,{"viewBox":f"0 0 {w} {h}","width":str(w),"height":str(h),
       "role":"img","aria-labelledby":"diagram-title diagram-desc",
       "font-family":"'Noto Sans CJK SC','Microsoft YaHei','PingFang SC',Arial,sans-serif"})
    def el(tag,attrs=None,text=None,parent=svg):
        e=ET.SubElement(parent,"{%s}%s"%(NS,tag),{k:str(v) for k,v in (attrs or {}).items()})
        if text is not None:e.text=text
        return e
    def text(x,y,value,size=17,color="#334155",weight=400,anchor="start",parent=svg):
        return el("text",{"x":x,"y":y,"font-size":size,"fill":color,"font-weight":weight,
                         "text-anchor":anchor},value,parent)
    el("title",{"id":"diagram-title"},spec["title"])
    el("desc",{"id":"diagram-desc"},"多智能体交互、可选视觉、原生模型、DriveHarness、独立评测与数据准入。状态标记区分已实现、部分接通与规划。")
    el("metadata",text=json.dumps({"date":spec["date"],"commit":spec["commit"],"notes":spec["notes"]},ensure_ascii=False))
    defs=el("defs")
    for key,(c,_) in COLORS.items():
        m=el("marker",{"id":"arrow-"+key,"viewBox":"0 0 10 10","refX":9,"refY":5,
          "markerWidth":7,"markerHeight":7,"orient":"auto-start-reverse","markerUnits":"strokeWidth"},parent=defs)
        el("path",{"d":"M 1 1 L 9 5 L 1 9 Z","fill":c},parent=m)
    el("rect",{"width":w,"height":h,"fill":"#FAFBFD"})
    el("rect",{"width":w,"height":164,"fill":"#132338"})
    el("rect",{"x":48,"y":39,"width":5,"height":66,"rx":2,"fill":"#54D3BC"})
    text(70,62,"DRIVERSI  /  SYSTEM ARCHITECTURE",15,"#93AECF",600)
    text(70,110,"DriveRSI 全系统架构",43,"#FFFFFF",700)
    text(70,142,"多智能体交互 · 可选写实视觉 · 模型训练接口 · 预算约束的 RSI 实验闭环",19,"#CFDBE9")
    text(1848,54,"SERVER 32341  ·  "+spec["date"],15,"#AEC1D8",500,"end")
    text(1848,80,"CODE "+spec["commit"]+"  /  接口状态快照",14,"#AEC1D8",400,"end")
    for x,(key,(color,label)) in zip([1210,1392,1662],STATUS.items()):
        el("circle",{"cx":x,"cy":119,"r":6,"fill":color})
        text(x+14,125,label,16,"#E3EBF4",500)
    for g in spec["groups"]:
        color,bg=COLORS[g["color"]]
        el("rect",{"x":g["x"],"y":g["y"],"width":g["width"],"height":g["height"],
                   "rx":16,"fill":bg,"stroke":"#DBE3EE","stroke-width":1})
        text(g["x"]+24,g["y"]+29,g["title"],22,color,700)
        text(g["x"]+24,g["y"]+51,g["subtitle"],15,"#596B82")

    for scene in spec.get("illustrations", []):
        panel=el("g",{"id":scene["id"],"transform":f"translate({scene['x']} {scene['y']})"})
        def shape(tag,attrs):
            return el(tag,attrs,parent=panel)
        def caption(x,y,value,size=16,color="#334155",weight=400,anchor="start"):
            return text(x,y,value,size,color,weight,anchor,parent=panel)
        def box(x,y,width,height,fill,stroke="none",radius=10):
            return shape("rect",{"x":x,"y":y,"width":width,"height":height,"rx":radius,"fill":fill,"stroke":stroke})
        def path(d,color,width=3,dash="",marker=""):
            a={"d":d,"fill":"none","stroke":color,"stroke-width":width,"stroke-linecap":"round","stroke-linejoin":"round"}
            if dash:a["stroke-dasharray"]=dash
            if marker:a["marker-end"]=f"url(#arrow-{marker})"
            return shape("path",a)
        def car(x,y,color,angle=0):
            g=el("g",{"transform":f"translate({x} {y}) rotate({angle})"},parent=panel)
            el("rect",{"x":-19,"y":-33,"width":38,"height":66,"rx":9,"fill":color,"stroke":"#FFFFFF","stroke-width":2},parent=g)
            el("path",{"d":"M -13 -20 L 13 -20 L 11 -8 L -11 -8 Z","fill":"#DBEAFE","opacity":.9},parent=g)
            el("rect",{"x":-12,"y":14,"width":24,"height":9,"rx":3,"fill":"#DBEAFE","opacity":.55},parent=g)
            el("path",{"d":"M -5 -26 L 0 -30 L 5 -26","fill":"none","stroke":"#FFFFFF","stroke-width":2},parent=g)
        box(0,0,scene["width"],scene["height"],"#FFFFFF","#C7D9E8",16)
        caption(24,31,scene["title"],24,"#173C58",700)
        caption(24,56,scene["subtitle"],15,"#62778C")
        for y,num,title,body in [
          (88,"01","每辆车都有自己的角色",["自车学习驾驶；他车执行不同策略。","普通交通和行人保留独立身份。"]),
          (194,"02","观察同一时刻，再共同推进",["各车分别决策，动作一起执行。","抢道或制动会改变后续响应。"]),
          (300,"03","保留失败，比较纠正",["从同一快照尝试不同动作。","高分候选仍需独立评分与准入。"])]:
            box(24,y,310,94,"#F1F6FB")
            box(38,y+14,34,25,"#DCE9F5",radius=6)
            caption(55,y+32,num,14,"#285A87",700,"middle")
            caption(81,y+32,title,17,"#234664",600)
            for i,line in enumerate(body):caption(39,y+58+i*22,line,14,"#5A7187")
        box(364,88,850,320,"#E9F0EA","#D8E4DC",14)
        for bx,by,bw,bh in [(390,125,130,44),(990,125,184,43),(385,326,196,50),(1020,328,150,48)]:
            box(bx,by,bw,bh,"#D2DDE6","#BACAD8",6)
            for k in range(int(bw//27)-1):
                box(bx+12+k*27,by+12,13,10,"#EDF3F7",radius=2)
        box(714,88,164,320,"#6C7C8D",radius=0)
        box(364,184,850,120,"#6C7C8D",radius=0)
        path("M 796 88 V 184 M 796 304 V 408","#F6D577",2,"10 9")
        path("M 364 244 H 714 M 878 244 H 1214","#F6D577",2,"10 9")
        path("M 724 88 V 176 M 868 88 V 176 M 724 312 V 408 M 868 312 V 408","#E0E8ED",2)
        path("M 364 194 H 706 M 886 194 H 1214 M 364 294 H 706 M 886 294 H 1214","#E0E8ED",2)
        for xx in range(733,863,22):
            box(xx,162,12,18,"#F8FBFD",radius=1)
            box(xx,309,12,18,"#F8FBFD",radius=1)
        for yy in range(202,291,18):box(891,yy,18,9,"#F8FBFD",radius=1)
        shape("path",{"d":"M 836 343 L 748 134 L 965 134 Z","fill":"#A78BFA","opacity":.20,
                      "stroke":"#8B5CF6","stroke-dasharray":"5 5","stroke-width":1.5})
        caption(919,154,"相机视野",14,"#7040BE",600)
        path("M 518 244 H 1090","#E59A24",4,marker="amber")
        path("M 836 365 C 836 310 836 267 792 244","#D65F64",4,"8 7")
        path("M 836 364 V 339 C 836 295 812 283 771 283 H 639 C 605 283 595 266 577 260","#149783",4,marker="teal")
        shape("circle",{"cx":792,"cy":244,"r":35,"fill":"#F9A8A8","fill-opacity":.22,
                        "stroke":"#F3B6B8","stroke-width":1.5,"stroke-dasharray":"4 4"})
        path("M 782 234 L 802 254 M 802 234 L 782 254","#FDD5D5",3)
        car(836,365,"#2673DB")
        car(540,244,"#D88A1C",90)
        car(1020,217,"#8CA2B5",-90)
        car(754,122,"#8CA2B5",180)
        caption(888,399,"自车 EGO",16,"#245FA7",700)
        caption(435,221,"对手 / 抢道",16,"#8A560D",700)
        box(914,260,140,28,"#FCE9EA",radius=6)
        caption(984,279,"风险分支",15,"#B64148",600,"middle")
        path("M 914 273 L 824 250","#C96D73",1.5)
        box(395,315,255,27,"#E4F6EF",radius=6)
        caption(522,334,"纠正候选：减速让行后转弯",14,"#167B68",600,"middle")
        for px,py in [(934,338),(957,346)]:
            shape("circle",{"cx":px,"cy":py-7,"r":4.5,"fill":"#C5709C"})
            path(f"M {px} {py-1} V {py+12} M {px-6} {py+5} H {px+6} M {px} {py+12} L {px-5} {py+20} M {px} {py+12} L {px+5} {py+20}","#B1608B",2)
        caption(1005,391,"行人 / 普通交通",13,"#60758A")
        caption(378,109,"俯视交互世界  ·  相机与物体共享同一状态",14,"#34536A",600)
        caption(364,426,"蓝：自车   橙：对手   灰：普通车辆   紫：相机视野   绿 / 红：候选分支",14,"#4E657A")
        caption(1244,108,"同一快照 → 两条后续",21,"#22415E",700)
        def camera(cx,positive):
            cy=128
            box(cx,cy,254,137,"#A8CEE6","#CAD6E3",8)
            box(cx,cy+67,254,70,"#4F5D68",radius=0)
            shape("path",{"d":f"M {cx+25} {cy+137} L {cx+105} {cy+67} H {cx+150} L {cx+239} {cy+137} Z","fill":"#60717F"})
            for bx,bw,bh in [(6,51,41),(199,48,50)]:
                box(cx+bx,cy+67-bh,bw,bh,"#9AAFBF",radius=2)
            path(f"M {cx+127} {cy+79} L {cx+124} {cy+96} M {cx+122} {cy+108} L {cx+119} {cy+133}","#EEE4B9",3)
            ax=cx+(167 if positive else 102);ay=cy+(68 if positive else 66)
            bw,bh=(38,25) if positive else (69,52)
            box(ax,ay,bw,bh,"#E2A03F","#F5CB81",4)
            box(ax+5,ay+4,bw-10,bh*.28,"#B9D7E7",radius=2)
            for dx in [7,bw-13]:box(ax+dx,ay+bh-3,7,6,"#263745",radius=1)
            box(cx+8,cy+8,68,21,"#17344A",radius=5)
            caption(cx+42,cy+23,"RGB 示意",11,"#FFFFFF",500,"middle")
            caption(cx+127,cy+161,scene["branch_positive" if positive else "branch_negative"],16,
                    "#167B68" if positive else "#BA4F53",600,"middle")
        camera(1244,True);camera(1530,False)
        path("M 1214 224 H 1232 V 194 H 1244","#7953CF",2,marker="purple")
        box(1244,311,540,76,"#F3F0FB",radius=10)
        caption(1262,338,"轨迹 + 多相机帧 + 行为结果",20,"#6240A6",700)
        caption(1262,364,"对齐 tick / 分支 / 相机 / 策略版本，再决定训练用途",15,"#756390")
        caption(1244,414,scene["camera_note"],13,"#8C759C")
        for i,label in enumerate(scene["steps"]):
            sx=24+i*356
            box(sx,448,326,30,"#EAF1F8",radius=7)
            caption(sx+163,469,label,16,"#36556E",600,"middle")
            if i<4:path(f"M {sx+331} 463 H {sx+349}","#5E7990",1.8,marker="slate")
        caption(24,494,"示意图用于解释机制，不代表实测 rollout、交通路权认证或合格的写实渲染效果。",12,"#7F8E9F")

    for e in spec["edges"]:
        pts=e["points"]
        assert len(pts)>=2 and all(len(p)==2 for p in pts)
        assert all(0<=p[0]<=w and 0<=p[1]<=h for p in pts)
        color=COLORS[e["color"]][0]
        attrs={"id":e["id"],"d":"M "+" L ".join(f"{x} {y}" for x,y in pts),
               "fill":"none","stroke":color,"stroke-width":2,
               "stroke-linejoin":"round","stroke-linecap":"round",
               "marker-end":"url(#arrow-"+e["color"]+")"}
        if e["dashed"]:attrs["stroke-dasharray"]="7 6"
        el("path",attrs)
    for n in spec["nodes"]:
        x=n["x"]-n["width"]/2;y=n["y"]-n["height"]/2
        assert x>=0 and y>=0 and x+n["width"]<=w and y+n["height"]<=h
        group=el("g",{"id":n["id"],"data-status":n["status"]})
        el("title",text=n["label"]+" — "+STATUS[n["status"]][1],parent=group)
        color,bg=COLORS[n["tone"]];st=STATUS[n["status"]][0]
        attrs={"x":x,"y":y,"width":n["width"],"height":n["height"],"rx":11,
               "fill":"#FFFFFF","stroke":"#D5DFEA","stroke-width":1.2}
        if n["status"]=="planned":attrs["stroke-dasharray"]="6 4"
        el("rect",attrs,parent=group)
        el("rect",{"x":x,"y":y+14,"width":4,"height":n["height"]-28,"rx":2,"fill":color},parent=group)
        el("circle",{"cx":x+n["width"]-17,"cy":y+21,"r":5.5,"fill":st},parent=group)
        text(x+17,y+31,n["label"],20,color,700,parent=group)
        for i,line in enumerate(n["body"]):
            text(x+17,y+57+i*25,line,16.5,"#334155",parent=group)
        if n["foot"]:
            text(x+17,y+n["height"]-13,n["foot"],12,"#75859A",parent=group)
    for e in spec["edges"]:
        if not e["label"]:continue
        label=e["label"]; width=sum(14 if ord(c)>0x2e80 else 7.5 for c in label)+16
        el("rect",{"x":e["lx"]-width/2,"y":e["ly"]-15,"width":width,"height":21,"rx":5,"fill":"#FAFBFD"})
        text(e["lx"],e["ly"],label,13,COLORS[e["color"]][0],500,"middle")
    text(48,h-39,"阅读方式：实线为主要数据流，虚线为可选输入、适配或反馈；模块右上角圆点表示接口状态。",15,"#53647B")
    text(48,h-15,"接口已实现不等于完整模型训练、实时渲染达标或工业认证。SVG / JSON 均可编辑，产物仅保存于服务器。",14,"#74849A")
    return ET.tostring(svg,encoding="unicode",xml_declaration=False)

def interaction_detail(spec, svg_text):
    source=ET.fromstring(svg_text)
    scene=next(x for x in spec["illustrations"] if x["id"]=="interaction-story")
    detail=ET.Element("{%s}svg"%NS,{"viewBox":f"0 0 1920 {scene['height']+48}",
        "width":"100%","height":"auto","style":"display:block;max-width:1920px;height:auto","role":"img",
        "aria-labelledby":"detail-title","font-family":source.attrib["font-family"]})
    ET.SubElement(detail,"{%s}title"%NS,{"id":"detail-title"}).text=scene["title"]
    ET.SubElement(detail,"{%s}rect"%NS,{"width":"1920","height":str(scene["height"]+48),"fill":"#FAFBFD"})
    detail.append(copy.deepcopy(source.find("{%s}defs"%NS)))
    group=copy.deepcopy(source.find(".//*[@id='interaction-story']"))
    group.set("transform","translate(48 24)")
    detail.append(group)
    return ET.tostring(detail,encoding="unicode")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("spec",type=Path);p.add_argument("--output",required=True,type=Path)
    args=p.parse_args()
    spec=json.loads(args.spec.read_text(encoding="utf-8"))
    result=build(spec);ET.fromstring(result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(result,encoding="utf-8")
    if spec.get("illustrations"):
        args.output.with_name("driversi-interaction-detail.svg").write_text(interaction_detail(spec,result),encoding="utf-8")
    print(json.dumps({"output":str(args.output),"nodes":len(spec["nodes"]),"edges":len(spec["edges"])},ensure_ascii=False))
if __name__=="__main__":main()
