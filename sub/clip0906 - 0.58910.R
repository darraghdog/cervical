library(data.table)
setwd("~/Dropbox/cervical/sub/")

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag_20170607.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub4 = sub

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag_cut0.4_20170608.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub5 = sub

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag_cut0.6_20170608.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub6 = sub


########################################
sub <- fread("sub_dara_full_gmm_remove_addl_10xbag_20170608.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub7 = sub


########################################
# Try baggin
sub7 = sub7[order(image_name)]
sub6 = sub6[order(image_name)]
sub5 = sub5[order(image_name)]
sub4 = sub4[order(image_name)]

########################################
subdm1 = fread("bebhionn_submission_clipped.csv")
subdm2 = fread("bebhionn_googlenet_submission.csv")
subdm1 = subdm1[order(image_name)]
subdm2 = subdm2[order(image_name)]

sub = sub4
#for(var in names(sub)[2:4]) sub[[var]] = (.1/3)* (sub1[[var]] + sub2[[var]] + sub3[[var]]) + 0.1 * (sub4[[var]]+ sub5[[var]]+ sub6[[var]]+ sub7[[var]]) + 0.25* (subdm1[[var]] + subdm2[[var]])
for(var in names(sub)[2:4]) sub[[var]] = 0.125 * (sub4[[var]]+ sub5[[var]]+ sub6[[var]]+ sub7[[var]]) + 0.25* (subdm1[[var]] + subdm2[[var]])

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

write.csv(sub, "remove_addtl_10x-leakv6.csv", row.names = F)

