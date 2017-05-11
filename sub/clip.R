library(data.table)
setwd("~/Dropbox/cervical/sub/")
sub <- fread("sub_dara_part_resnet_raw_5xbag_20170510.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub
write.csv(sub, "sub_dara_part_resnet_raw_5xbag_20170510_clipped.csv", row.names = F)
