from modules.sharemodule import os,dt,np,pd,plt,pdist,o3d
from modules import fileread as fr
from modules import preprocess as pp
from modules import tools as t
import math
import modules.fileconverter as fc

# background_las = "C:/Users/o2d/git/LiDAR/LiDAR/Python/traindata/lasdata/April/24_with_iso/background/3000ms/Frame.las"
# background_pcd = "C:/Users/o2d/git/LiDAR/LiDAR/Python/traindata/pcddata/April/24_with_iso/background/3000ms/Frame.pcd"
# [background, now] = fr.las2pcd(background_las,background_pcd)
# start = dt.datetime.now()
# background_pcd = fr.ReadPCD(background)
# background_pcd = pp.SelectPCD(background_pcd,xlim=[],ylim=[],zlim=[-1.3,0.6])#天井、地面削除

# lasdir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/traindata/lasdata/April/24_with_iso/two_walker/200ms/"

#1458あたりを一気に重畳すれば背景取れそ
# def FileSyn(pcddir,outputpath):
#     tflist = []
#     count = {}
#     judge = 0
#     lv1list = []
#     lv2list = []
#     for name in os.listdir(pcddir):
#         if not name[-8:] in count.keys():
#             # tflist.append(name[-8:])
#             count[name[-8:]]=0
#             # print(tflist)
#             print(count)
#         else:
#             count[name[-8:]] += 1
#     print(count)
#     for key in count:
#         if count.get(key) > 1:
#             tflist.append(key)
#     # print(tflist)


#     for i in range(sum(os.path.isfile(os.path.join(pcddir, name)) for name in os.listdir(pcddir))-1):
#         lv=os.listdir(pcddir)[i]
#         if lv[-8:] == "0000.pcd":
#             judge = 1
#             print(lv)
#             print("judge changed")
#         if judge == 1:
#             if lv[-8:]==tflist[0]:
#                 pcd = fr.ReadPCD(pcddir+lv)
#                 lv1list.append(pcd)
#             elif lv[-8:]==tflist[1]:
#                 pcd = fr.ReadPCD(pcddir+lv)
#                 lv2list.append(pcd)
#     print("finish")
#     print(f"len of 1:{len(lv1list)}")
#     print(f"len of 2:{len(lv2list)}")
#     for j in range(len(lv1list)):
#         pcd = pp.pcdsummarize(lv1list[j],lv2list[j],angle=180, corrected_x=4.45,corrected_y=0)
#         if len(str(j))==1:
#             o3d.io.write_point_cloud(outputpath + "/frame_" + "000" + str(j) + ".pcd",pcd)
#             print(outputpath + "/frame_" + "000" + str(j) + ".pcd")
#         elif len(str(j))==2:
#             o3d.io.write_point_cloud(outputpath + "/frame_" + "00" + str(j) + ".pcd",pcd)
#             print(outputpath + "/frame_" + "00" + str(j) + ".pcd")
#         elif len(str(j))==3:
#             o3d.io.write_point_cloud(outputpath + "/frame_" + "0" + str(j) + ".pcd",pcd)
#             print(outputpath + "/frame_" + "0" + str(j) + ".pcd")
#         elif len(str(j))>3:
#             o3d.io.write_point_cloud(outputpath + "/frame_" + str(j) + ".pcd",pcd)
#             print(outputpath + "/frame_" + str(j) + ".pcd")
        

#     # for i in range(sum(os.path.isfile(os.path.join(pcddir, name)) for name in os.listdir(pcddir))-1):
#     #     print("---------------")
#     #     lv1=os.listdir(pcddir)[i]
#     #     lv2=os.listdir(pcddir)[i+1]
#     #     print(f"lv1:{lv1}")
#     #     print(f"lv2:{lv2}")
#     #     if lv1[-8:]==tflist[0] and lv2[-8:]==tflist[1]:
#     #         #重畳の処理
#     #         pcd1 = fr.ReadPCD(pcddir+lv1)
#     #         pcd2 = fr.ReadPCD(pcddir+lv2)
#     #         pcd = pp.pcdsummarize(pcd1,pcd2,angle=math.pi,corrected_x=4.45,corrected_y=0)
#     #         print(i)
#     #         print(outputpath)
#     #         print(f"lv1:{lv1}")
#     #         print(f"lv2:{lv2}")
#     #         # if i>999:
#     #         #     pcddf = t.o3dtoDataFrame(pcd)
#     #         #     print(pcddf['x'].max())
#     #         #     print(pcddf['y'].max())
#     #         #     print(pcddf['x'].min())
#     #         #     print(pcddf['y'].min())
#     #         #     o3d.visualization.draw_geometries([pcd],str(i))
#     #         # print(f"{i}")
#     #         if len(str(i))==1:
#     #             o3d.io.write_point_cloud(outputpath + "/frame_" + "000" + str(i) + ".pcd",pcd)
#     #         elif len(str(i))==2:
#     #             o3d.io.write_point_cloud(outputpath + "/frame_" + "00" + str(i) + ".pcd",pcd)
#     #         elif len(str(i))==3:
#     #             o3d.io.write_point_cloud(outputpath + "/frame_" + "0" + str(i) + ".pcd",pcd)
#     #         elif len(str(i))>3:
#     #             o3d.io.write_point_cloud(outputpath + "/frame_" + str(i) + ".pcd",pcd)
#     #         # i+=1#同じファイルを重畳させないため
#     #         # lv1=os.listdir(pcddir)[i]
#     #         # lv2=os.listdir(pcddir)[i+1]
#     #         # print(f"lv1:{lv1}")
#     #         # print(f"lv2:{lv2}")

#     print("finish")

#twowalker
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/twowalker/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/synthesized_twowalker",angle=180,x=4.6,y=0)

#big
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/big/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/synthesized_big",angle=180,x=4.6,y=0)

# 20230609
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/09/20230609_1person/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/09/1person_sync",angle=180,x=5,y=0)
# background
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/09/20230609_background/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/09/background_sync",angle=180,x=5,y=0)

# 20230620
# background
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/20/background/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/piled/June/20/background/",angle=180,x=4.545,y=0)
# background_camera
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/20/background_camera/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/background_camera/",angle=180,x=4.545,y=0)
# chair
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/20/chair/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/chair/",angle=180,x=4.5,y=0)
# ishi2
pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/20/ishi2/"
fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/ishi2/",angle=180,x=4.5,y=0)
# iso2
# pcddir = "C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/June/20/iso2/"
# fc.FileSyn(pcddir=pcddir,outputpath="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/syned/June/20/iso2/",angle=180,x=4.545,y=0)
# fc.csv2pcd(1000,read_csv_path="C:/Users/o2d/git/LiDAR/LiDAR/Python/csvdata/April/24_with_iso/two_livox.csv",output_folder_path="C:/Users/o2d/git/LiDAR/LiDAR/Python/pcddata/April/24_with_iso/twowalker_fromcsv")
# fc.Lidarsyn()