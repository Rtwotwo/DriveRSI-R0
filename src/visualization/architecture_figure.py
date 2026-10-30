"""Editable eight-stage RSI overview. Mini-visuals are schematic, never results."""

from html import escape
from urllib.parse import quote

STEPS = ('sft', 'evaluate', 'diagnose', 'retrieve', 'generate', 'index', 'skill', 'retrain')
BLUE, TEAL, AMBER, PURPLE = '#2864bd', '#078779', '#bb7516', '#7957b1'


def render_overview(system):
    rows = system.get('steps', [])
    if tuple(r['id'] for r in rows) != STEPS:
        raise ValueError('Overview requires the declared eight-stage execution order')
    out = ['''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 1460 1040" role="img" aria-labelledby="figure-title figure-desc">
<title id="figure-title">DriveRSI 八步训练与系统改进闭环</title>
<desc id="figure-desc">从初始SFT、评测、短板记录、跨划分检索到历史扰动物理执行、DriveRL TTS未来、GS合成、独立派生库、技能记忆和下一轮训练。图形为机制示意，能力状态与实验结果分开。点击步骤查看说明。</desc>
<defs>
<marker id="arrow-blue" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0 0L7 3L0 6Z" fill="#2864bd"/></marker>
<marker id="arrow-teal" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0 0L7 3L0 6Z" fill="#078779"/></marker>
<marker id="arrow-purple" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto" markerUnits="strokeWidth"><path d="M0 0L7 3L0 6Z" fill="#7957b1"/></marker>
</defs>
<style>text{font-family:Inter,"Noto Sans CJK SC","Microsoft YaHei",Arial,sans-serif;fill:#203c50}a{cursor:pointer}a:hover .node{stroke:#299b8a;stroke-width:2.5}a:focus .node{stroke:#299b8a;stroke-width:3}.muted{fill:#607b8d}.white{fill:#fff}.tiny{font-size:12px}.label{font-size:14px}.body{font-size:15px}.title{font-size:20px;font-weight:700}.line{fill:none;stroke-width:2}.node{fill:#fff;stroke:#d6e4ed;stroke-width:1.3}</style>
<rect width="1460" height="1040" fill="#f6f9fc"/>
<path d="M0 0H1460V80H0Z" fill="#152e42"/>
<path d="M40 20V61" stroke="#56cfb1" stroke-width="4"/>
<text x="58" y="31" class="tiny" style="fill:#abd1e1">DRIVERSI / MODEL LEARNING + IMPROVEMENT LEARNING</text>
<text x="58" y="60" class="title white" style="font-size:27px">八步闭环 · 模型变强，改进方法也要变强</text>
<text x="1080" y="32" class="label" style="fill:#c4d9e5">目标流程 / 非实验结果</text>
<text x="1080" y="58" class="tiny" style="fill:#c4d9e5">点击步骤查看输入、执行、产物与验收</text>
''']

    def text(x, y, label, size=15, color=None, weight=None, **attrs):
        extra = ''.join(f' {escape(str(k))}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
        style = f'font-size:{size}px' + (f';fill:{color}' if color else '') + (f';font-weight:{weight}' if weight else '')
        out.append(f'<text x="{x}" y="{y}" style="{style}"{extra}>{escape(str(label))}</text>')

    def rect(x, y, w, h, fill='#fff', stroke='#d8e5ec', radius=7, **attrs):
        extra = ''.join(f' {k}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
        out.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" fill="{fill}" stroke="{stroke}"{extra}/>')

    def path(d, color, width=2, dash=None, marker=None, **attrs):
        extra = ''.join(f' {k}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        if marker:
            extra += f' marker-end="url(#arrow-{marker})"'
        out.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{extra}/>')

    def dot(x, y, radius=4, fill=TEAL):
        out.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{fill}"/>')

    def car(x, y, color, scale=1):
        out.append(f'<g transform="translate({x} {y}) scale({scale})">')
        rect(-19, -9, 38, 18, color, '#ffffff', 4)
        rect(-10, -6, 6, 12, '#d7e9ee', 'none', 1)
        path('M11 -6V6', '#eef8fa', 2)
        out.append('</g>')

    def frame(x, y, w=112, h=46, variant=0):
        rect(x, y, w, h, '#dcebf1', '#c9dce5', 5)
        out.append(f'<path d="M{x} {y+h}L{x+w*.37} {y+16}H{x+w*.65}L{x+w} {y+h}Z" fill="#8498a7"/>')
        path(f'M{x+w*.5} {y+23}V{y+h-4}', '#fbf4ce', 1.6, '5 4')
        rect(x+12, y+5, 19, 19, '#b1c9d3', 'none', 1)
        rect(x+w-29, y+9, 20, 16, '#b7d0c1', 'none', 2)
        car(x+w*.57+variant, y+h*.73, '#c58c36', .31)

    # Sequential artifacts and two different feedback loops.
    edges = [('sft','evaluate','M350 224H390'),('evaluate','diagnose','M700 224H740'),
             ('diagnose','retrieve','M1050 224H1090'),('retrieve','generate','M1245 312V352'),
             ('generate','index','M1245 670V718'),('index','skill','M1090 812H1050'),
             ('skill','retrain','M740 812H700')]
    for src, dest, d in edges:
        path(d, BLUE, 2.1, marker='blue', **{'data-edge-type':'sequence','data-from':src,'data-to':dest})
    path('M545 906V926H20V108H545V136', TEAL, 2, '7 5', 'teal', **{'data-edge-type':'model-feedback','data-from':'retrain','data-to':'evaluate'})
    text(58, 101, '模型回路：候选权重返回评测，合格才晋级', 13, TEAL)
    path('M895 906V950H1430V224H1400', PURPLE, 2, '7 5', 'purple', **{'data-edge-type':'system-feedback','data-from':'skill','data-to':'retrieve'})
    path('M545 312V330H1068V690H895V718', PURPLE, 1.6, '5 5', 'purple', **{'data-edge-type':'utility-feedback','data-from':'evaluate','data-to':'skill'})
    text(655, 324, '配对复评：收益 / 退化；无对照不记收益', 12, PURPLE)
    text(630, 975, '系统回路：收益 / 成本 → 技能记忆 → 后续策略；选择效率提升仍待验证', 15, PURPLE, 600)
    text(70, 1015, 'PDMS：p∈[0,1]；界面分值为100p。完整物理分支、原生评分时域和模型监督窗口分别验收。', 13, '#5d788c')

    positions = [(40,136,176,BLUE),(390,136,176,BLUE),(740,136,176,AMBER),(1090,136,176,AMBER),
                 (1090,352,318,TEAL),(1090,718,188,PURPLE),(740,718,188,PURPLE),(390,718,188,TEAL)]
    for i,(row,(x,y,h,color)) in enumerate(zip(rows,positions)):
        target = 'index.html#view=architecture&amp;step='+quote(row['id'],safe='')
        out.append(f'<a href="{target}" xlink:href="{target}" target="_top" data-step="{row["id"]}" aria-label="第{i+1}步 {escape(row["title"],quote=True)}">')
        rect(x,y,310,h,**{'class':'node'})
        path(f'M{x+1} {y+2}V{y+h-2}',color,3)
        dot(x+26,y+32,15, '#edf4fa' if i<4 else '#edf6f2')
        text(x+17,y+37,f'{i+1:02d}',13,color,700)
        text(x+51,y+38,row['short'],19,color,700)
        text(x+18,y+65,row['subtitle'],14,'#637f91')
        state = {'pilot':'小规模已接通','partial':'部分接通 / 待验收','active':'当前维护','planned':'计划接入'}[row['status']]
        dot(x+19,y+h-18,3.5,TEAL if row['status']=='pilot' else AMBER)
        text(x+31,y+h-14,state,12,'#627d8e')
        if i==0:
            for j in range(3):frame(x+20+j*5,y+91-j*4,68,43,j)
            path(f'M{x+109} {y+114}H{x+140}',BLUE,1.8,marker='blue')
            for j in range(3):rect(x+155+j*12,y+89+j*6,46,38,'#e6effa','#83a6d3',4)
            text(x+229,y+112,'M₀',24,BLUE,600)
        elif i==1:
            for j,v in enumerate([6,35,51,42,14,48,39,55]):rect(x+22+j*22,y+143-v,12,v,'#dfad62' if j in [0,4] else '#82abc8','none',2)
            text(x+215,y+107,'逐场景',13,BLUE)
            text(x+215,y+129,'子指标',13,BLUE)
        elif i==2:
            rect(x+19,y+88,111,52,'#fff0e8','#ecd6c8',6)
            text(x+48,y+119,'p = 0',20,'#ac5542',700)
            rect(x+145,y+88,145,52,'#fff7e5','#ecdfc3',6)
            text(x+158,y+119,'0 < p < 0.6',18,AMBER,700)
        elif i==3:
            for j in range(6):dot(x+28+j%3*15,y+101+j//3*18,4,BLUE if j%2 else PURPLE)
            text(x+82,y+108,'全库视觉排序',13,'#627d8e')
            text(x+82,y+130,'语义簇另作对照',13,'#627d8e')
            path(f'M{x+185} {y+116}H{x+213}',AMBER,1.7,marker='blue')
            rect(x+221,y+91,72,48,'#eaf4ee','#bcd9ca',5)
            text(x+228,y+110,'navtrain',12,TEAL,700)
            text(x+233,y+130,'Top-3',13,TEAL)
        elif i==4:
            text(x+19,y+91,'0–2 s LQR → 2–7 s 原生TTS',14,TEAL,600)
            for j in range(4):frame(x+20+(j%2)*139,y+111+(j//2)*58,129,46,j-2)
            text(x+21,y+236,'轨迹准入 p > 0.95 + ADE',16,TEAL,700)
            text(x+21,y+260,'源位姿PSNR：按实验配置',15,TEAL,600)
            text(x+21,y+282,'来源绑定；不等于新视角质量保证',12,'#627d8e')
        elif i==5:
            for j in range(3):
                rect(x+20+j*90,y+88,80,50,'#f1ecf8','#d9cdea',6)
                text(x+29+j*90,y+109,['新图像向量','文本结构簇','分支血缘'][j],13,PURPLE,600)
                path(f'M{x+30+j*90} {y+123}H{x+77+j*90}', '#c3b4da',3)
            text(x+22,y+157,'合成库独立 · hist / future / logs',13,'#627d8e')
        elif i==6:
            labels=['适用条件','执行程序','验收规则','收益 / 成本']
            for j,s in enumerate(labels):
                xx=x+22+(j%2)*136; yy=y+85+(j//2)*34
                rect(xx,yy,125,28,'#f0ebf7','#dfd4ee',5);text(xx+10,yy+19,s,13,PURPLE)
            text(x+22,y+158,'记住处理方法，不只记住一个样本',13,'#627d8e')
        else:
            rect(x+20,y+87,115,49,'#edf4f5','#d0e0e6',5);text(x+31,y+108,'普通续训',14,'#4c7182',600);text(x+31,y+127,'原始数据',12,'#627d8e')
            rect(x+151,y+87,138,49,'#e9f5ef','#b9d9cc',5);text(x+162,y+108,'扩充组',14,TEAL,600);text(x+162,y+127,'原始 + 合成增量',12,'#627d8e')
            text(x+22,y+157,'同父模型 · 同预算 · 回到第2步',13,TEAL)
        out.append('</a>')

    # Harness is cross-cutting; it is not a ninth sequential stage.
    rect(40,352,310,318,'#ecf3f8','#d6e4ed',9)
    text(60,387,'DriveHarness 贯穿全程',20,'#285873',700)
    text(60,416,'每一步都绑定同一条证据链',14,'#627d8e')
    items=[('选题','能力条件、检索证据与场景多样性'),('执行','教师 / 渲染 / 训练的技能程序'),('准入','原生监督、分支和像素身份核对'),('成本','预算、失败、重试和实际耗时'),('反馈','收益、退化、拒绝与记忆更新')]
    for j,(label,value) in enumerate(items):
        yy=452+j*39;dot(66,yy-5,3,TEAL);text(80,yy,label,14,'#285873',700);text(122,yy,value,12,'#59758a')
    text(60,651,'当前规则调度；学习选择策略仍待验证',13,'#285873')

    # World-to-observation mechanism, not a measured scene or a physical claim.
    rect(390,352,660,318,'#ffffff','#d6e4ed',9)
    text(410,383,'同一物理分支：新历史 → 锚点 → 纠正未来',19,'#285873',700)
    text(410,407,'道路与参与者位置决定相机观测；示意图不代表实际合格轨迹',12,'#627d8e')
    rect(410,420,620,116,'#e5efea','#dbe7e2',7)
    rect(412,443,616,73,'#738a99','none',0)
    path('M420 478H1020','#f0e2aa',2,'12 9')
    for j in range(5):rect(429+j*30,425,18,10,'#c4d9de','none',2)
    for j in range(5):dot(850+j*33,431,6,'#b7d4bc')
    path('M434 496H586',BLUE,3)
    for xx in [442,482,522,562]:dot(xx,496,5,'#bed8f6')
    path('M588 496H720',AMBER,2,'6 5')
    path('M590 496C630 496 647 457 690 457H776C815 457 826 496 865 496H1005',TEAL,3,marker='teal')
    car(590,496,BLUE,.95);car(725,496,'#ce943e',.9);car(873,458,'#bdcfdc',.9)
    path('M895 458H958',AMBER,1.8,marker='blue')
    text(439,532,'新历史',12,BLUE,600);text(631,435,'同状态纠正未来',12,TEAL,600);text(909,532,'其他交通响应',12,'#627d8e')
    text(410,560,'四相机 × (4 新历史 + 10 未来)；PDMS评2–6 s，实际执行至7 s',13,'#285873',600)
    x0=430; dx=40
    path(f'M{x0} 600H{x0+14*dx}', '#bdd0db',2)
    for i in range(15):
        xx=x0+i*dx; is_history=1<=i<=4
        if i>0:
            rect(xx-11,582,22,24,'#cfe1f5' if is_history else '#c6e7da','none',4, **{'data-window':'history' if is_history else 'future','data-time':i/2})
            text(xx-4,599,'H' if is_history else 'F',11,BLUE if is_history else TEAL,600)
        else:path(f'M{xx} 581V607', '#7d94a5',2,'3 3')
        text(xx-10,625,f'{i/2:g}',11,'#627d8e')
    path('M590 571V633',AMBER,1.5,'4 3');text(555,651,'锚点 2 s',12,AMBER,600)
    text(710,651,'未来图像仅作监督，不能进入当前历史输入',12,'#627d8e')

    rect(40,718,310,188,'#eef5f7','#d6e4ed',9)
    text(60,753,'执行加速组件',19,'#285873',700)
    text(60,785,'常驻GS · 源对照缓存 · 四相机批量',13,'#627d8e')
    text(60,818,'无损存储 · 常驻导出与新特征服务',13,'#627d8e')
    text(60,851,'独立多路基准已验证；生产调度待接',13,'#627d8e')
    text(60,884,'当前课程只展示真实阶段与预算状态',12,'#627d8e')
    out.append('</svg>')
    return '\n'.join(out)
