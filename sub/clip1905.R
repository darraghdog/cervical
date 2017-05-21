library(data.table)
setwd("~/Dropbox/cervical/sub/")
sub <- fread("sub_dara_part_resnet_raw_5xbag_20170510.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub1 = sub
# write.csv(sub, "sub_dara_part_resnet_raw_5xbag_20170510_clipped.csv", row.names = F)

########################################
sub <- fread("sub_dara_full_resnet_dmcrop_5xbag_20170513.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub2 = sub

########################################
# Try bagging
sub2 = sub2[order(image_name)]
sub1 = sub1[order(image_name)]


########################################
subdm1 = fread("bebhionn_submission_clipped.csv")
subdm2 = fread("bebhionn_googlenet_submission.csv")
subdm1 = subdm1[order(image_name)]
subdm2 = subdm2[order(image_name)]

sub = sub1
for(var in names(sub)[2:4]) sub[[var]] = 0.5* (sub1[[var]] + sub2[[var]]) + 0.5* (subdm1[[var]] + subdm2[[var]])
par(mfrow=c(2,2))
for(var in names(sub)[2:4]) hist(sub[[var]], main = paste(var, "submission"), ylim = c(0,250))
sub
sub1
sub2
subdm1
subdm2

write.csv(sub, "7030_dhdmbest.csv", row.names = F)

