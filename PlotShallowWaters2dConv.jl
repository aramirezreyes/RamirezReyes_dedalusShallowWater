using HDF5,  Printf, Makie, Statistics, PyCall

post = pyimport("dedalus.tools.post")
subprocess = pyimport("subprocess")

a = "analysis_convgravicor_bigdt_s"
length(a)

function readoutput(exp_name,dir="./")
    file_list = list_files(exp_name,dir)
    dirtoread = string(dir,exp_name,"/")    
    c = h5open(string(dirtoread,file_list[1]), "r") do file
        read(file)
    end
    y       = c["scales"]["y"]["1.0"]
    x       = c["scales"]["x"]["1.0"];
    times   = c["scales"]["sim_time"]
    h       = c["tasks"]["h"];
    #centers = c["tasks"]["conv_centers"];
    #centers_times = c["tasks"]["conv_centers_times"];
    u       = c["tasks"]["u"];
    v       = c["tasks"]["v"];
    for i in 2:length(file_list)
        c = h5open(string(dirtoread,file_list[i]), "r") do file
            read(file)
        end
        times   = cat(times,c["scales"]["sim_time"],dims=1) 
        h       = cat(h,c["tasks"]["h"],dims=3)
        #centers = cat(centers,c["tasks"]["conv_centers"],dims=3)
        #centers_times = cat(centers_times,c["tasks"]["conv_centers_times"],dims=3)
        u       = cat(u,c["tasks"]["u"],dims=3)
        v       = cat(v,c["tasks"]["v"],dims=3)
    end
    return x,y,times,h,u,v
end

function list_files(exp_name,dir="./")
    dirtoread = string(dir,exp_name,"/")
    file_prefix = string(exp_name,"_s")
    file_list = filter(x -> endswith(x, ".h5"), readdir(dirtoread))
    sort!(file_list,by = x -> tryparse(Int,x[length(file_prefix)+1:end-3]));
    return file_list
end

function merge_dedalus_output(exp_name)
    print(subprocess.check_output("find $(exp_name)", shell=true).decode());
    post.merge_process_files(exp_name,cleanup=true)
end

#file_list = list_files()

#x,y,times,h,u,v = readoutput(file_list);

# t=1200
# title = @sprintf "Convecting. Time: %1.2f days" times[t]/86400
# p1 = plot(1e-3x,1e-3y,h[:,:,t],st=:heatmap,xlabel="X (Km)",ylabel="Y (Km)",
#     zlabel = "H(m)", title="Height")
# #p2 = plot(1e-3x,1e-3y,centers[:,:,t],st=:heatmap,border=nothing,xlabel="X (Km)",ylabel="Y (Km)",
# #    title = title,clims = (0.0,0.5))
# p3 = plot(1e-3x,1e-3y,(u.*u+v.*v)[:,:,t],st=:heatmap,border=nothing,xlabel="X (Km)",ylabel="Y (Km)",
#     title = title,colorbar=true)
# #p4 = plot(1e-3x,1e-3y,centers_times[:,:,t],st=:heatmap,border=nothing,xlabel="X (Km)",ylabel="Y (Km)",
# #        title = title,colorbar=true)
#plot(p1,p3)

# function animateoutput()
#     anim = @animate for t in 1:1:size(h,3)
#         title = @sprintf "Convecting. Time: %1.2f days" times[t]/86400
#         p1 = plot(1e-3x,1e-3y,h[:,:,t],c=:viridis,st=:heatmap,xlabel="X (Km)",ylabel="Y (Km)",
#                   zlabel = "H(m)", title="Height")
#         #p2 = plot(1e-3x,1e-3y,centers[:,:,t],st=:heatmap,border=nothing,xlabel="X (Km)",ylabel="Y (Km)",
#         #    title = title,clims = (0.0,0.5))
#         p3 = plot(1e-3x,1e-3y,(u.*u+v.*v)[:,:,t],st=:heatmap,border=nothing,xlabel="X (Km)",ylabel="Y (Km)",
#                   title = title,colorbar=true)
#         #p4 = plot(1e-3x,1e-3y,centers_times[:,:,t],st=:heatmap,border=nothing,xlabel="X (Km)",ylabel="Y (Km)",
#         #        title = title,colorbar=true)
#         plot(p1,p3)
#     end
#     mp4(anim, "./anim_fps17.mp4", fps = 10)
# end
function animatesurface(every=1)
    scene = surface(h[:,:,1])
    surf = scene[end]
    eyepos = Vec3f0(200, 200, 100)
    lookat = Vec3f0(0)
    AbstractPlotting.update_cam!(scene, eyepos, lookat)
    record(scene, "output.mp4", 1:every:size(h,3),framerate = 10) do t
        surf[1] = h[:,:,t]
        AbstractPlotting.update_cam!(scene, eyepos, lookat)
        AbstractPlotting.update_limits!(scene)
        AbstractPlotting.update!(scene)
    end
    return scene
end


function testanimate()
    h = u = v = rand(10,10,100);
    x = y = 1:10
    t = 1:100
    animateheatmap(h,u,v,x,y,t)
end


function animateheatmap2(h,u,v,x,y,times,every=1,output="output.mp4")
    times_days = times./86400
    sp = sqrt.(u.*u .+ v.*v)
    scene1 = heatmap(1e-3*x,1e-3*y,h[:,:,1],interpolate=false,colorrange=(39.9,40.1),colormap=:pu_or)
    heat = scene1[end]
    colorbar1 = colorlegend(heat,camera=campixel!,raw=true)
    #scene1[Axis][:ticks][:textsize] = (3,3)
    #scene1[Axis][:names][:textsize] = (3,3)
    scene2 = heatmap(1e-3*x,1e-3*y,sp[:,:,1],interpolate=false,colorrange=(0.0,2.0))
    speed = scene2[end]
    #scene2[Axis][:ticks][:textsize] = (3,3)
    #scene2[Axis][:names][:textsize] = (3,3)
    colorbar2 = colorlegend(speed,camera=campixel!,raw=true)
    sc1t = title(scene1,"Height (m)")
    sc2t = title(scene2,"Speed (m/s)")
    scene3 = vbox(vbox(sc1t,colorbar1),vbox(sc2t,colorbar2))
    sct3 = title(scene3,"Time: 0s",parent=Scene(resolution=(1200,570)))
    record(sct3, "output.mp4", 1:every:size(h,3)) do t
        titlestr = @sprintf("Time: %1.1f days",times_days[t])
        heat[3] = h[:,:,t]
        speed[3] = sp[:,:,t]
        push!(sct3.children[2][end][:text],titlestr)
        AbstractPlotting.update_limits!(scene1)
        AbstractPlotting.update!(scene1)
        AbstractPlotting.update_limits!(scene2)
        AbstractPlotting.update!(scene2)
        AbstractPlotting.update!(scene3)
    end
end

function animateheatmap(h,u,v,x,y,times,every=1,output="output.mp4")
    times_days = times./86400
    
    #    colormap_symmetric = ::pu_or
    colormap_symmetric = :bluesreds
    sp = sqrt.(u.*u .+ v.*v)
    max_value_sp = maximum(sp)
    h_anomaly = h .- mean(h,dims=(1,2))
    max_value_hano = maximum(abs.(h_anomaly))
    scene1 = heatmap(1e-3*x,1e-3*y,h_anomaly[:,:,1],interpolate=false,colorrange=(0.0,max_value_hano))
    heat = scene1[end]
    colorbar1 = colorlegend(heat,camera=campixel!,raw=true)
   #########
    scene2 = heatmap(1e-3*x,1e-3*y,sp[:,:,1],interpolate=false,colorrange=(0.0,max_value_sp))
    speed = scene2[end]
    colorbar2 = colorlegend(speed,camera=campixel!,raw=true)
#########
    scene3 = heatmap(1e-3*x,1e-3*y,u[:,:,1],interpolate=false,colorrange=(-1max_value_sp,max_value_sp),
                     colormap=colormap_symmetric)
    speedu = scene3[end]
    colorbar3 = colorlegend(speedu,camera=campixel!,raw=true)
    #########
    scene4 = heatmap(1e-3*x,1e-3*y,v[:,:,1],interpolate=false,colorrange=(-1max_value_sp,max_value_sp),
                     colormap=colormap_symmetric)
    speedv = scene4[end]
    colorbar4 = colorlegend(speedv,camera=campixel!,raw=true)
    #########
    sc1t = title(scene1,"Height (m)")
    sc2t = title(scene2,"Speed (m/s)")
    sc3t = title(scene3,"U (m/s)")
    sc4t = title(scene4,"V (m/s)")
    grandscene = hbox( vbox(vbox(sc3t,colorbar3),vbox(sc4t,colorbar4)) ,
                   vbox(vbox(sc1t,colorbar1),vbox(sc2t,colorbar2)))
    grandsct = title(grandscene,"Time: 0s",parent=Scene(resolution=(770,800)))
    record(grandsct, "output.mp4", 1:every:size(h,3)) do t
        titlestr = @sprintf("Time: %1.1f days",times_days[t])
        heat[3] = h_anomaly[:,:,t]
        speed[3] = sp[:,:,t]
        speedu[3] = u[:,:,t]
        speedv[3] = v[:,:,t]
        push!(grandsct.children[2][end][:text],titlestr)
#        AbstractPlotting.update_limits!(scene1)
        AbstractPlotting.update!(scene1)
 #       AbstractPlotting.update_limits!(scene2)
        AbstractPlotting.update!(scene2)
        AbstractPlotting.update!(scene3)
        AbstractPlotting.update!(scene4)
        AbstractPlotting.update!(grandscene)
    end
end



function readandanimate2(exp_name)
    x,y,times,h,u,v = readoutput(exp_name)
    animateheatmap2(h,u,v,x,y,times)
end





function readandanimate(exp_name)
    x,y,times,h,u,v = readoutput(exp_name)
    animateheatmap(h,u,v,x,y,times)
end




function test()
    scene = heatmap(rand(10,10))
    dat = scene[end]
    record(scene, "test.mp4", 1:10) do t
        dat[1] = rand(10,10)
        AbstractPlotting.update_limits!(scene)
        AbstractPlotting.update_limits!(scene)
    end
end

