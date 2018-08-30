import cv2,os
from xml.etree import ElementTree as ET
from lxml import etree, objectify
import random
imgsdir="DETRAC-train-data"
annosdir="DETRAC-Train-Annotations-XML"

vocannodir="Annotations" 


def showgt(bconv2voc=True,bshow=False):
    imglist=os.listdir(imgsdir)
    for i in range(len(imglist)):
        il=imglist[i]
        print(i,il)
        imgdir=imgsdir+"/"+il
        annopath=annosdir+"/"+il+".xml"
        vocannosubdir=vocannodir+"/"+il
        if bconv2voc and not os.path.exists(vocannosubdir):
            os.makedirs(vocannosubdir)
        annoxml=ET.parse(annopath)
        ignored_region=annoxml.find("ignored_region")
        iboxes=ignored_region.findall("box")
        files=os.listdir(imgdir)
        frames=annoxml.findall("frame")
        maped_frames={}
        for frame in frames:
            frame_num=(int)(frame.get("num"))
            maped_frames[frame_num-1]=frame
        for j in range(len(files)):
            file=files[j]
            #imgpath=imgdir+"/"+file
            filename="img%05d"%(j+1)
            imgpath=imgdir+"/"+filename+".jpg"
            img=cv2.imread(imgpath)
            if j in maped_frames:
                frame=maped_frames[j]
                targets=frame.find("target_list").findall("target")
                if bshow:
                    for target in targets:
                        id=target.get("id")
                        bbox=target.find("box")
                        vehicle_type=target.find("attribute").get("vehicle_type")
                        speed=target.find("attribute").get("speed")
                        left=(int)((float)(bbox.get("left")))
                        top=(int)((float)(bbox.get("top")))
                        width=(int)((float)(bbox.get("width")))
                        height=(int)((float)(bbox.get("height")))
                        pt1=(left,top)
                        pt2=(left+width,top+height)
                        type_id=vehicle_type+":"+id
                        cv2.putText(img,type_id,pt1,1,1,(0,255,0))
                        cv2.putText(img,speed,(left,top+30),1,1,(0,0,255))
                        cv2.rectangle(img,pt1,pt2,(255,0,0))
                    for ib in iboxes:
                        left=(int)((float)(ib.get("left")))
                        top=(int)((float)(ib.get("top")))
                        width=(int)((float)(ib.get("width")))
                        height=(int)((float)(ib.get("height")))
                        pt1=(left,top)
                        pt2=(left+width,top+height)
                        cv2.rectangle(img,pt1,pt2,(0,0,0),2)#,cv2.FILLED
                    cv2.putText(img,str(j),(0,20),3,1,(0,0,255))
                    cv2.imshow("img",img)
                

                if bconv2voc:
                    vocannopath=vocannosubdir+"/"+filename+".xml"
                    E = objectify.ElementMaker(annotate=False)
                    anno_tree = E.annotation(
                        E.folder('0'),
                        E.filename(il+"/"+os.path.basename(imgpath)),
                        E.source(
                            E.database('car'),
                            E.annotation('VOC'),
                            E.image('CK')
                        ),
                        E.size(
                            E.width(img.shape[1]),
                            E.height(img.shape[0]),
                            E.depth(img.shape[2])
                        ),
                        E.segmented(0)
                    )
                    count=0
                    for target in targets:
                        vehicle_type=target.find("attribute").get("vehicle_type")
                        if not vehicle_type=="car":
                            continue
                        count+=1
                        bbox=target.find("box")
                        speed=target.find("attribute").get("speed")
                        left=(int)((float)(bbox.get("left")))
                        top=(int)((float)(bbox.get("top")))
                        width=(int)((float)(bbox.get("width")))
                        height=(int)((float)(bbox.get("height")))
                        E2 = objectify.ElementMaker(annotate=False)
                        anno_tree2 = E2.object(
                            E.name(vehicle_type),
                            E.pose(),
                            E.truncated("0"),
                            E.difficult(0),
                            E.bndbox(
                                E.xmin(left),
                                E.ymin(top),
                                E.xmax(left+width),
                                E.ymax(top+height)
                            )
                        )
                        anno_tree.append(anno_tree2)
                    if count>0:
                        etree.ElementTree(anno_tree).write(vocannopath, pretty_print=True)
            #cv2.waitKey(1)

def split_train_test(train_ratio=0.8):
    all_images=[]
    subdirs=os.listdir(vocannodir)
    for sub in subdirs:
        subdir=vocannodir+"/"+sub
        files=os.listdir(subdir)
        for file in files:
            all_images.append(sub+"/"+file[:-4])
    random.shuffle(all_images)
    with open("ImageSets/Main/train.txt","w")as ftrain:
        with open("ImageSets/Main/val.txt","w")as fval:
            for i in range(len(all_images)):
                if i <train_ratio*len(all_images):
                    ftrain.write(all_images[i]+"\n")
                else:
                    fval.write(all_images[i]+"\n")
    
if __name__=="__main__":
    showgt()
    split_train_test()