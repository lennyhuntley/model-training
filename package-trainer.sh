rm -f trainer.tar trainer.tar.gz
tar cvf trainer.tar package
gzip trainer.tar
gsutil cp trainer.tar.gz gs://know-now-app-trainer-lh/know-now-app-trainer.tar.gz