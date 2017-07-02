df.b = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn/results.txt")
df.b.unif = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn-unif/results.txt")
df.b.elr = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn2/results.txt") # equal lrs for both gen/disc
df.b.bb1 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn-bb1/results.txt")

par(mfrow=c(1,2))
plot(df.b$discriminator_loss,type="l",ylim=c(0,10)); 
lines(df.b.unif$discriminator_loss,col="red"); lines(df.b.elr$discriminator_loss,col="blue"); 
lines(df.b.bb1$discriminator_loss,col="purple")
plot(df.b$generator_loss,type="l",ylim=c(0,10)); 
lines(df.b.unif$generator_loss,col="red");
lines(df.b.elr$generator_loss,col="blue");  
lines(df.b.bb1$generator_loss,col="purple")

# perhaps the generator was too strong for the disc. for gr1,
# so let's try something else...

par(mfrow=c(1,1))

df.b.bb2 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn-bb2/results.txt")
df.b.bb2b = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn-bb2b/results.txt")


plot(df.b.bb1$discriminator_loss,type="l",ylim=c(0,3))
lines(df.b.bb2$discriminator_loss,col="red")
lines(df.b.bb2b$discriminator_loss,col="blue")

plot(df.b.bb1$generator_loss,type="l",ylim=c(0,5))
lines(df.b.bb2$generator_loss,col="red")
lines(df.b.bb2b$generator_loss,col="blue")

df.bb.i1.repeat = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_repeat/results.txt")
df.bb.i1 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1/results.txt")

df.bb.i1.skip = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip/results.txt")
df.bb.i1.skip2 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip2/results.txt")
df.bb.i1.skip3 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip3/results.txt")
df.bb.i1.skip4 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip4/results.txt")
df.bb.i1.skip5 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip5/results.txt")
df.bb.i1.skip6 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip6/results.txt")


plot(df.bb.i1$generator_loss,type="l",xlim=c(0,300)); 
lines(df.bb.i1.repeat$generator_loss,col="red");  
lines(df.bb.i1.skip$generator_loss,col="blue")
lines(df.bb.i1.skip2$generator_loss,col="purple")
lines(df.bb.i1.skip5$generator_loss,col="green")
lines(df.bb.i1.skip6$generator_loss,col="orange")

plot(df.bb.i1$discriminator_loss,type="l",xlim=c(0,300))
lines(df.bb.i1.repeat$discriminator_loss,col="red")
lines(df.bb.i1.skip$discriminator_loss,col="blue")
lines(df.bb.i1.skip2$discriminator_loss,col="purple")
lines(df.bb.i1.skip5$discriminator_loss,col="green")
lines(df.bb.i1.skip6$discriminator_loss,col="orange")

df.bb.i1.skip6ls = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip6ls/results.txt")
df.bb.i1.skip6lseq = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip6lseq/results.txt")
df.bb.i1.skip6ls2k = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip6ls_ld2000/results.txt")
df.bb.i1.skip7 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip7ls_rmsprop/results.txt")
df.bb.i1.skip7d5 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1_skip7lsd5_rmsprop/results.txt")


plot(df.bb.i1.skip6ls$generator_loss,type="l"); lines(df.bb.i1.skip6lseq$generator_loss,col="red"); lines(df.bb.i1.skip7$generator_loss,col="blue")
plot(df.bb.i1.skip6ls$discriminator_loss,type="l"); lines(df.bb.i1.skip6lseq$discriminator_loss,col="red"); lines(df.bb.i1.skip7$discriminator_loss,col="blue")

df.bb.i1.ls = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1ls_rmspropd/results.txt")
df.bb.i1.ls.weakd = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1ls_rmspropd_weakd/results.txt")
df.bb.i1.ls.weakd2 = read.csv("~/Desktop/lisa_tmp4_4/nasa_depth2terrain/dcgan/output/gan-heightmap-ld1000-b-discbn_i1ls_rmspropd_weakd2/results.txt")

plot(df.bb.i1.ls$discriminator_loss,type="l"); lines(df.bb.i1.ls.weakd$discriminator_loss,col="red"); lines(df.bb.i1.ls.weakd2$discriminator_loss,col="blue")
plot(df.bb.i1.ls$generator_loss,type="l"); lines(df.bb.i1.ls.weakd$generator_loss,col="red"); lines(df.bb.i1.ls.weakd2$generator_loss,col="blue")

tmp = read.csv("~/Desktop/lisa_tmp4_4/forward_thinking/output/fw_basic_net_1_sigm.txt")
