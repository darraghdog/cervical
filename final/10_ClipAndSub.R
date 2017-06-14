library(data.table)
setwd("~/Dropbox/cervical/sub/")

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub4 = sub

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag_cut0.4.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub5 = sub

########################################
sub <- fread("sub_dara_full_remove_addl_10xbag_cut0.6.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub6 = sub


########################################
sub <- fread("sub_dara_full_gmm_remove_addl_10xbag.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub7 = sub

########################################
sub <- fread("output.csv")

mat = as.matrix(sub[,-1,with=F])
mat[mat < 0.02] = 0.02
mat = mat/rowSums(mat)

cols= names(sub)[-1]
for(i in 1:3) sub[[cols[i]]] = mat[,i]
sub8 = sub

########################################
# Try baggin
sub8 = sub8[order(image_name)]
sub7 = sub7[order(image_name)]
sub6 = sub6[order(image_name)]
sub5 = sub5[order(image_name)]
sub4 = sub4[order(image_name)]

sub = sub4
for(var in names(sub)[2:4]) sub[[var]] = (0.125) * (sub4[[var]]+ sub5[[var]]+ sub6[[var]]+ sub7[[var]]) + 0.5*(sub8[[var]]) # + 0.15* (subdm1[[var]] + subdm2[[var]])

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

write.csv(sub, "submission2.csv", row.names = F)

