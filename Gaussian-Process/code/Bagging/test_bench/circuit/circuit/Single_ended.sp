.option post
.option ingold=2
.option numdgt=10
.option measdgt=10
.option dcon=1
.inc param
.inc opamp.sp

.param vdd_v=2
.param vin_cm='0.5*vdd_v'
.param vin_low='0.375*vdd_v'
.param vin_high='0.625*vdd_v'
.param period=100e-6
.param risetime=1e-9
.param falltime=1e-9

xac vin_ac+ vin_ac- vo_ac vdd_ac vss opamp
vdd_ac vdd_ac 0 vdd_v
e+ vin_ac+ 101 100 0 0.5
e- vin_ac- 101 100 0 -0.5
vcm 101 0 dc=vin_cm
vs 100 0 dc=0 ac=1

xtr vin_tr+ vo_tr   vo_tr vdd_tr vss opamp
vpulse vin_tr+ 0 pulse (vin_low vin_high 0 risetime falltime 'period/2' period)
vdd_tr vdd_tr 0 vdd_v

vss vss 0 0

.dc vdd_ac vdd_v vdd_v vdd_v
.meas dc iq find '1e6*abs(i(vdd_ac))' at vdd_v

.ac dec 100 1 50Meg

.meas ac gain find vdb(vo_ac) at=1
.meas ac ugf_actual when vdb(vo_ac)=0
.meas ac phase min vp(vo_ac) from=1 to=ugf_actual
.meas ac gm   find '-1*vdb(vo_ac)'   when   vp(vo_ac)=-178

.meas ac ugf=param('ugf_actual * 1e-6')
.meas ac pm=param('phase + 180')


.trans 'period/10000' 'period*1'
.meas tran vomax max v(vo_tr)
.meas tran vomin min v(vo_tr)
.meas tran srr_actual deriv v(vo_tr) when v(vo_tr)='vomin + 0.5 * (vomax - vomin)' rise=1
.meas tran srf_actual deriv v(vo_tr) when v(vo_tr)='vomin + 0.5 * (vomax - vomin)' fall=1
.meas tran srr=param('abs(srr_actual)/1e6')
.meas tran srf=param('abs(srf_actual)/1e6')

.end
