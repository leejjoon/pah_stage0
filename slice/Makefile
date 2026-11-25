neutral:
	./run_neutral.sh
neutral2:
	./run_neutral_v2.sh
# version 3 (neutrals + cations): `spectra` on the GH200 node, `regress DIR=<tables>` on an x86-64 machine
nc3-spectra:
	./run_nc_v3.sh spectra
nc3-regress:
	./run_nc_v3.sh regress $(DIR)
.PHONY: neutral neutral2 nc3-spectra nc3-regress
