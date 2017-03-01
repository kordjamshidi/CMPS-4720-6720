library(R.matlab)
setwd("E:/research/data/sz")
fraw <- readMat("fMRI_preproc.mat")
summary(fraw)

fmri <- fraw$im.sorted

#block is the match of voxel to ROI#
block <- data.frame(t(unlist(fraw$roiClass)))
colnames(block) <- c("block")

##extract ROI data##
tfmri <- data.frame(t(fmri))
brain <- cbind(block,tfmri)
level <- factor(brain$block)
brain <- data.frame(level,tfmri)
# tapply(brain[,2],brain$level,mean)
brain_mean <- sapply(brain[,2:209], function(x) tapply(x,brain$level,mean))

ROI_patient <- t(brain_mean[,1:92])
ROI_control <- t(brain_mean[,93:208])