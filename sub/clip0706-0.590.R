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
sub <- fread("sub_dara_part_resnet_raw_5xbag_20170521.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub3 = sub

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag_20170607.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub4 = sub


for (i in 2:4) print(cor(sub1[[i]], sub2[[i]]))

########################################
# Try baggin
sub4 = sub4[order(image_name)]
sub3 = sub3[order(image_name)]
sub2 = sub2[order(image_name)]
sub1 = sub1[order(image_name)]
for(i in 1:3) print(cor(sub1[[i+1]], sub2[[i+1]]))

########################################
subdm1 = fread("bebhionn_submission_clipped.csv")
subdm2 = fread("bebhionn_googlenet_submission.csv")
subdm1 = subdm1[order(image_name)]
subdm2 = subdm2[order(image_name)]

sub = sub1
for(var in names(sub)[2:4]) sub[[var]] = 0.125* (sub1[[var]] + sub2[[var]] + sub3[[var]] + sub4[[var]]) + 0.25* (subdm1[[var]] + subdm2[[var]])

#########################################
# Load leak
dupes = fread("../features/dupes_leak.csv", skip = 1)
dupes = dupes[,c(2,4), with=F]
setnames(dupes, c("image_name", "act"))
dupes

sub[image_name %in% dupes[act=="Type_1"]$image_name, `:=`(Type_1 = 0.98, Type_2 = 0.02, Type_3 = 0.02)]
sub[image_name %in% dupes[act=="Type_2"]$image_name, `:=`(Type_1 = 0.02, Type_2 = 0.98, Type_3 = 0.02)]
sub[image_name %in% dupes[act=="Type_3"]$image_name, `:=`(Type_1 = 0.02, Type_2 = 0.02, Type_3 = 0.98)]
sub[image_name %in% dupes$image_name]
dupes
#View(sub)

write.csv(sub, "remove_addtl_10x-leak.csv", row.names = F)

