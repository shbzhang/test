set_plot,'ps'
restore,'layerlist.aic'
device,filename='AICer.ps',/color,/encap,bits_per=8,xsize=22,ysize=22
aicer_main,layerlist,pageoffset=[2,2],singlepanel=0
aicer_main,layerlist,pageoffset=[12,3],singlepanel=0,pagescale=0.7
device,/close
set_plot,'x'
end
