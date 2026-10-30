"""Deterministically create a single-file SVG contact sheet from actual gallery data."""
from __future__ import annotations
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

NS='http://www.w3.org/2000/svg'
ET.register_namespace('',NS)
ROOT=Path(__file__).resolve().parents[3]
CONFIG=[
 ('multiview','e064658636885f06__virtual_front','01  七车共同交互 / 非 ego 车辆视角',
  '规则策略群体；7 辆受控车，各主体独立观察后联合推进。',
  '相机属于 e064…5f06；新机位、参与者外观与高程质量尚未准入。'),
 ('pittsburgh-recovery','CAM_F0','02  Pittsburgh / DriveRL 教师恢复分支',
  'DriveRL TTS；扰动对照发生碰撞，教师反事实 PDM 98.28 / 100。',
  '轨迹准入通过；两处路障外观未绑定，图像仍只用于诊断。'),
 ('boston-control-search','CAM_F0','03  Boston / 有界控制搜索纠正',
  '9 个控制候选中选择 candidate-07；反事实 PDM 94.74 → 95.18 / 100。',
  '当前日志对象与重建节点身份匹配为 0；保留残影和配准问题，不作为训练数据。')]

def node(parent,tag,**attrs):
    return ET.SubElement(parent,f'{{{NS}}}{tag}',{k.replace('_','-'):str(v) for k,v in attrs.items()})
def text(parent,x,y,value,size=22,color='#243c50',weight='400',**attrs):
    n=node(parent,'text',x=x,y=y,fill=color,font_size=size,font_weight=weight,**attrs);n.text=value;return n
def path(parent,points,color,width=2,dash=None,opacity=1):
    if len(points)<2:return
    d='M '+' L '.join(f'{x:.3f},{y:.3f}' for x,y in points)
    attrs={'d':d,'fill':'none','stroke':color,'stroke_width':width,'stroke_linecap':'round','stroke_linejoin':'round','opacity':opacity}
    if dash:attrs['stroke_dasharray']=dash
    return node(parent,'path',**attrs)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def generate(root:Path,output:Path):
    viz=root/'viz';svg=ET.Element(f'{{{NS}}}svg',{'width':'2000','height':'2190','viewBox':'0 0 2000 2190','style':'width:100%;height:auto;display:block','role':'img','aria-labelledby':'title description'})
    title=node(svg,'title',id='title');title.text='DriveRSI：同一物理时刻的多主体 BEV 与真实生成的 GS 图像'
    desc=node(svg,'desc',id='description');desc.text='三个封存案例，同 tick 30、3.0秒。原始合成图像保持像素内容，BEV由对应实际物理状态生成。并非学习型对抗自博弈或合格训练图像证明。'
    style=node(svg,'style');style.text='text{font-family:"Noto Sans CJK SC","Microsoft YaHei","PingFang SC",sans-serif} '
    defs=node(svg,'defs');evidence=[]
    node(svg,'rect',width=2000,height=2190,fill='#edf2f7')
    node(svg,'rect',width=2000,height=155,fill='#10273b')
    node(svg,'rect',x=42,y=30,width=6,height=91,fill='#35c5b3')
    text(svg,70,55,'DRIVERSI / ACTUAL ROLLOUT & RENDERED VIEWS',20,'#a8c4d8','600')
    text(svg,70,103,'多主体交互轨迹 × 同时刻 BEV × 合成视角',40,'#ffffff','700')
    text(svg,70,135,'实际封存案例 · 每行均为 tick 30 / 3.0 s · 图像原样保留 · 服务器 32341',20,'#c7d8e6')
    for row,(cid,cam_id,heading,sub,note) in enumerate(CONFIG):
        top=180+row*640;case_file=viz/'data'/f'{cid}.json';data=json.loads(case_file.read_text());frame=next(x for x in data['frames'] if x['tick']==30)
        camera=next(c for c in frame['cameras'] if c['id']==cam_id)
        owner=next(a for a in frame['actors'] if a['id']==camera['owner_actor_id'])
        assert owner['present'] and abs(frame['time_seconds']-3)<1e-9
        img=(viz/camera['image']).resolve();assert img.is_relative_to(viz.resolve()) and img.is_file()
        evidence.append({'case_id':cid,'data':str(case_file),'data_sha256':sha(case_file),'image':str(img),'image_sha256':sha(img),'tick':30,'time_seconds':3,'owner_actor_id':owner['id'],'camera_id':cam_id})
        node(svg,'rect',x=30,y=top,width=1940,height=620,rx=18,fill='#ffffff',stroke='#d5e0e9')
        text(svg,54,top+38,heading,29,'#163b53','700')
        text(svg,54,top+69,sub,21,'#526b7e')
        text(svg,950,top+38,'BEV 与 RGB 严格同 tick',19,'#087a6b','600')
        node(svg,'rect',x=1700,y=top+16,width=240,height=34,rx=17,fill='#fff1cf')
        text(svg,1720,top+40,'诊断图像 / trainable=false',17,'#80581c','600')
        x,y,w,h=54,top+88,904,508.5
        clip=node(defs,'clipPath',id=f'bev-{row}');node(clip,'rect',x=x,y=y,width=w,height=h,rx=8)
        node(svg,'rect',x=x,y=y,width=w,height=h,rx=8,fill='#080e14',stroke='#273b4a')
        g=node(svg,'g',clip_path=f'url(#bev-{row})')
        scale=w/110.;cx=owner['x'];cy=owner['y']
        def project(pt):return (x+w/2+(float(pt[0])-cx)*scale,y+h/2-(float(pt[1])-cy)*scale)
        # Metric, map-aligned crop. X/Y use exactly the same scale.
        for gx in range(int(cx//10)*10-60,int(cx//10)*10+71,10):path(g,[project((gx,cy-50)),project((gx,cy+50))],'#192732',1)
        for gy in range(int(cy//10)*10-40,int(cy//10)*10+51,10):path(g,[project((cx-65,gy)),project((cx+65,gy))],'#192732',1)
        for road in data['roads']:path(g,[project(p) for p in road],'#667d8c',1.2,opacity=.85)
        for route in data.get('routes',[]):
            if route.get('actor_id') in (owner['id'],'ego'):path(g,[project(p) for p in route['points']],'#748d9d',2.5,'10 7',.8)
        for tr in data.get('trajectories',[]):
            role=tr['role'];aid=tr['actor_id'];is_owner=aid==owner['id'];is_ego=aid=='ego'
            if role=='executed':
                path(g,[project(p) for p in tr['points']],'#30d1bd' if is_owner or is_ego else '#bc9b64',4 if is_owner or is_ego else 1.4,opacity=1 if is_owner or is_ego else .5)
            elif is_ego and role=='baseline':path(g,[project(p) for p in tr['points']],'#dc7041',3,'12 6')
            elif is_ego and role=='reference':path(g,[project(p) for p in tr['points']],'#6585b6',2,'2 6',.75)
        for a in frame['actors']:
            if not a['present']:continue
            co='#2780d9' if a['controlled'] else '#e4b561'
            if a['kind']=='pedestrian':co='#ba5d93'
            if a['kind']=='cyclist':co='#9067bc'
            if a['kind']=='other_object':co='#7f8b94'
            if a.get('collision'):co='#d64b4b'
            c=math.cos(a['yaw']);s=math.sin(a['yaw']);pts=[]
            for lx,ly in [(a['length']/2,a['width']/2),(a['length']/2,-a['width']/2),(-a['length']/2,-a['width']/2),(-a['length']/2,a['width']/2)]:
                pts.append(project((a['x']+lx*c-ly*s,a['y']+lx*s+ly*c)))
            node(g,'polygon',points=' '.join(f'{xx:.3f},{yy:.3f}' for xx,yy in pts),fill=co,stroke='#ffffff',stroke_width=1.2)
            centre=project((a['x'],a['y']));front=project((a['x']+a['length']*.38*c,a['y']+a['length']*.38*s))
            path(g,[centre,front],'#ffffff',1.8)
            if a['id']==owner['id']:
                node(g,'circle',cx=centre[0],cy=centre[1],r=max(13,a['length']*.65*scale),fill='none',stroke='#c589ff',stroke_width=3)
                text(g,centre[0]+22,centre[1]-20,'相机主体',18,'#ddaaff','700')
        # Scale and north directions are exact map axes, not camera orientation.
        node(svg,'rect',x=x+16,y=y+h-48,width=210,height=34,rx=8,fill='#101e2a',opacity='.95')
        path(svg,[(x+30,y+h-28),(x+30+10*scale,y+h-28)],'#a5bdce',3)
        text(svg,x+40+10*scale,y+h-21,'10 m',17,'#a5bdce','600')
        text(svg,x+18,y+28,'结构化 BEV · 仅显示当前在场对象',19,'#b7cfdf','600')
        text(svg,x+w-65,y+29,'Y ↑',18,'#b7cfdf','600')
        rx,ry,rw,rh=1010,y,904,508.5
        node(svg,'rect',x=rx,y=ry,width=rw,height=rh,rx=8,fill='#0b1721')
        encoded=base64.b64encode(img.read_bytes()).decode('ascii')
        node(svg,'image',x=rx,y=ry,width=rw,height=rh,href='data:image/png;base64,'+encoded,preserveAspectRatio='xMidYMid meet')
        node(svg,'rect',x=rx+14,y=ry+12,width=rw-28,height=34,rx=6,fill='#102334',opacity='.83')
        camera_label='非 ego / 虚拟前视相机' if camera.get('virtual') else 'ego / 前视相机 CAM_F0'
        text(svg,rx+28,ry+36,'GS 原始合成图像 · '+camera_label+' · 3.0 s',19,'#ffffff','600')
        text(svg,54,top+612,note,18,'#8d5b29')
    footer=2120
    text(svg,54,footer,'图例',20,'#1b3d52','700')
    for px,label,color,dash in [(125,'路线','#748d9d','10 7'),(330,'实际执行','#0caa94',None),(570,'扰动对照','#dc7041','12 6'),(805,'日志参考','#6585b6','2 6')]:
        path(svg,[(px,footer-7),(px+58,footer-7)],color,4,dash);text(svg,px+70,footer,label,19)
    text(svg,1060,footer,'蓝：受控车辆   黄：背景车辆   紫圈：当前相机主体',19)
    text(svg,54,2155,'来源：同一案例的封存世界状态与渲染结果；轨迹可叠加未来执行，仅用于展示，不是策略输入。',18,'#536b7e')
    text(svg,54,2182,'规则群体 / 固定扰动 / 教师搜索不等同于学习型对抗 Self-Play；反事实 PDM 不是官方 navtest 成绩。',18,'#536b7e')
    m=node(svg,'metadata');m.text=json.dumps({'kind':'actual_case_contact_sheet','version':1,'cases':evidence,'new_rendering':False,'image_editing':False,'time_seconds':3.0},ensure_ascii=False)
    ET.indent(svg,space='  ');output.parent.mkdir(parents=True,exist_ok=True)
    raw=ET.tostring(svg,encoding='utf-8',xml_declaration=True);output.write_bytes(raw)
    ET.fromstring(raw)
    print(json.dumps({'output':str(output),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'embedded_images':3,'cases':3},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path)
    a=p.parse_args();generate(a.root,a.output or a.root/'viz/preview.svg')
