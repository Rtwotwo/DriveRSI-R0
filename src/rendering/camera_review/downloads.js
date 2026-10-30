/* Static, pre-encoded videos: downloading never records browser playback. */
(function(root){
  let manifest=null, selected=null;
  const labels={source:'下载原日志白膜 MP4',branch:'下载新轨迹白膜 MP4',comparison:'下载左右对比 MP4'};
  function link(file,label){const a=document.createElement('a');a.href=file.path;a.download=file.path.split('/').pop();a.className='videoDownload';a.textContent=label+' · '+(file.bytes/1048576).toFixed(1)+' MB';return a;}
  function show(){const host=document.getElementById('videoLinks');if(!host||!manifest)return;host.replaceChildren();
    const files=manifest.videos.filter(v=>v.camera_id===selected);
    for(const kind of ['source','branch','comparison']){const v=files.find(f=>f.variant===kind);if(v)host.append(link(v,labels[kind]));}
    const note=document.createElement('small');note.textContent=files.length===1?'虚拟车辆没有原日志机位，仅提供新轨迹白膜。':'原日志与新轨迹分别使用各自标定；对比视频按同一时间轴排列。';host.append(note);
    document.getElementById('videoCamera').textContent=selected||'选择相机';
  }
  root.WhiteboxDownloads={select(id){selected=id;show();}};
  fetch('videos.json').then(r=>{if(r.status===404)return null;if(!r.ok)throw Error('视频清单加载失败');return r.json();}).then(d=>{
    if(!d){document.getElementById('videoStatus').textContent='此场景尚未导出白膜视频。';return;}
    if(d.kind!=='camera_whitebox_videos'||!Array.isArray(d.videos))throw Error('视频清单格式不兼容');
    manifest=d;document.getElementById('videoStatus').textContent=`${d.scene_duration_seconds.toFixed(2)} 秒 · 按日志时间播放 · ${d.output_fps} fps 封装（重复帧保持，不生成新观测） · 地图图层为导出时默认配置`;
    document.getElementById('videoArchive').append(link(d.archive,'下载全部白膜视频 ZIP'));
    const details=document.createElement('a');details.href='videos.json';details.textContent='视频来源与时间记录';document.getElementById('videoArchive').append(details);show();
  }).catch(e=>{document.getElementById('videoStatus').textContent=e.message;});
})(globalThis);
