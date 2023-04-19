/* Created by Language version: 7.7.0 */
/* VECTORIZED */
#define NRN_VECTORIZED 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "scoplib_ansi.h"
#undef PI
#define nil 0
#include "md1redef.h"
#include "section.h"
#include "nrniv_mf.h"
#include "md2redef.h"
 
#if METHOD3
extern int _method3;
#endif

#if !NRNGPU
#undef exp
#define exp hoc_Exp
extern double hoc_Exp(double);
#endif
 
#define nrn_init _nrn_init__Ih_linearized_v2_frozen
#define _nrn_initial _nrn_initial__Ih_linearized_v2_frozen
#define nrn_cur _nrn_cur__Ih_linearized_v2_frozen
#define _nrn_current _nrn_current__Ih_linearized_v2_frozen
#define nrn_jacob _nrn_jacob__Ih_linearized_v2_frozen
#define nrn_state _nrn_state__Ih_linearized_v2_frozen
#define _net_receive _net_receive__Ih_linearized_v2_frozen 
 
#define _threadargscomma_ _p, _ppvar, _thread, _nt,
#define _threadargsprotocomma_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt,
#define _threadargs_ _p, _ppvar, _thread, _nt
#define _threadargsproto_ double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt
 	/*SUPPRESS 761*/
	/*SUPPRESS 762*/
	/*SUPPRESS 763*/
	/*SUPPRESS 765*/
	 extern double *getarg();
 /* Thread safe. No static _p or _ppvar. */
 
#define t _nt->_t
#define dt _nt->_dt
#define gIhbar _p[0]
#define ehcn _p[1]
#define V_R _p[2]
#define ihcn _p[3]
#define wInf _p[4]
#define wTau _p[5]
#define phi _p[6]
#define wAlpha _p[7]
#define a1 _p[8]
#define a2 _p[9]
#define a3 _p[10]
#define b1 _p[11]
#define b2 _p[12]
#define b3 _p[13]
#define foo _p[14]
#define dwinf _p[15]
#define wBeta _p[16]
#define v _p[17]
#define _g _p[18]
 
#if MAC
#if !defined(v)
#define v _mlhv
#endif
#if !defined(h)
#define h _mlhh
#endif
#endif
 
#if defined(__cplusplus)
extern "C" {
#endif
 static int hoc_nrnpointerindex =  -1;
 static Datum* _extcall_thread;
 static Prop* _extcall_prop;
 /* external NEURON variables */
 /* declaration of user functions */
 static int _mechtype;
extern void _nrn_cacheloop_reg(int, int);
extern void hoc_register_prop_size(int, int, int);
extern void hoc_register_limits(int, HocParmLimits*);
extern void hoc_register_units(int, HocParmUnits*);
extern void nrn_promote(Prop*, int, int);
extern Memb_func* memb_func;
 
#define NMODL_TEXT 1
#if NMODL_TEXT
static const char* nmodl_file_text;
static const char* nmodl_filename;
extern void hoc_reg_nmodl_text(int, const char*);
extern void hoc_reg_nmodl_filename(int, const char*);
#endif

 extern void _nrn_setdata_reg(int, void(*)(Prop*));
 static void _setdata(Prop* _prop) {
 _extcall_prop = _prop;
 }
 static void _hoc_setdata() {
 Prop *_prop, *hoc_getdata_range(int);
 _prop = hoc_getdata_range(_mechtype);
   _setdata(_prop);
 hoc_retpushx(1.);
}
 /* connect user functions to hoc names */
 static VoidFunc hoc_intfunc[] = {
 "setdata_Ih_linearized_v2_frozen", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gIhbar_Ih_linearized_v2_frozen", "S/cm2",
 "ehcn_Ih_linearized_v2_frozen", "mV",
 "V_R_Ih_linearized_v2_frozen", "mV",
 "ihcn_Ih_linearized_v2_frozen", "mA/cm2",
 0,0
};
 /* connect global user variables to hoc */
 static DoubScal hoc_scdoub[] = {
 0,0
};
 static DoubVec hoc_vdoub[] = {
 0,0,0
};
 static double _sav_indep;
 static void nrn_alloc(Prop*);
static void  nrn_init(_NrnThread*, _Memb_list*, int);
static void nrn_state(_NrnThread*, _Memb_list*, int);
 static void nrn_cur(_NrnThread*, _Memb_list*, int);
static void  nrn_jacob(_NrnThread*, _Memb_list*, int);
 /* connect range variables in _p that hoc is supposed to know about */
 static const char *_mechanism[] = {
 "7.7.0",
"Ih_linearized_v2_frozen",
 "gIhbar_Ih_linearized_v2_frozen",
 "ehcn_Ih_linearized_v2_frozen",
 "V_R_Ih_linearized_v2_frozen",
 0,
 "ihcn_Ih_linearized_v2_frozen",
 "wInf_Ih_linearized_v2_frozen",
 "wTau_Ih_linearized_v2_frozen",
 "phi_Ih_linearized_v2_frozen",
 0,
 0,
 0};
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 19, _prop);
 	/*initialize range parameters*/
 	gIhbar = 1e-05;
 	ehcn = -45;
 	V_R = 0;
 	_prop->param = _p;
 	_prop->param_size = 19;
 
}
 static void _initlists();
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _Ih_linearized_v2_frozen_reg() {
	int _vectorized = 1;
  _initlists();
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 19, 0);
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 Ih_linearized_v2_frozen /home/meowlin/projects/LFP_kernel_files/LFPykernels-main/examples/mod/Ih_linearized_v2_frozen.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
 {
   a1 = 0.001 * 6.43 ;
   a2 = 154.9 ;
   a3 = 11.9 ;
   b1 = 0.001 * 193.0 ;
   b2 = 33.1 ;
   wAlpha = a1 * ( V_R + a2 ) / ( exp ( ( V_R + a2 ) / a3 ) - 1.0 ) ;
   wBeta = b1 * exp ( V_R / b2 ) ;
   wInf = wAlpha / ( wAlpha + wBeta ) ;
   foo = wAlpha / ( a3 * a1 ) * exp ( ( V_R + a2 ) / a3 ) + ( V_R + a2 ) / b2 - 1.0 ;
   wTau = 1.0 / ( wAlpha + wBeta ) ;
   dwinf = - wBeta * wAlpha * wTau * wTau / ( V_R + a2 ) * foo ;
   phi = gIhbar * dwinf * ( ehcn - V_R ) ;
   }

}
}

static void nrn_init(_NrnThread* _nt, _Memb_list* _ml, int _type){
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; double _v; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 v = _v;
 initmodel(_p, _ppvar, _thread, _nt);
}
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   ihcn = gIhbar * wInf * ( v - ehcn ) - wInf * phi ;
   }
 _current += ihcn;

} return _current;
}

static void nrn_cur(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; double _rhs, _v; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml]; _ppvar = _ml->_pdata[_iml];
#if CACHEVEC
  if (use_cachevec) {
    _v = VEC_V(_ni[_iml]);
  }else
#endif
  {
    _nd = _ml->_nodelist[_iml];
    _v = NODEV(_nd);
  }
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
 	}
 _g = (_g - _rhs)/.001;
#if CACHEVEC
  if (use_cachevec) {
	VEC_RHS(_ni[_iml]) -= _rhs;
  }else
#endif
  {
	NODERHS(_nd) -= _rhs;
  }
 
}
 
}

static void nrn_jacob(_NrnThread* _nt, _Memb_list* _ml, int _type) {
double* _p; Datum* _ppvar; Datum* _thread;
Node *_nd; int* _ni; int _iml, _cntml;
#if CACHEVEC
    _ni = _ml->_nodeindices;
#endif
_cntml = _ml->_nodecount;
_thread = _ml->_thread;
for (_iml = 0; _iml < _cntml; ++_iml) {
 _p = _ml->_data[_iml];
#if CACHEVEC
  if (use_cachevec) {
	VEC_D(_ni[_iml]) += _g;
  }else
#endif
  {
     _nd = _ml->_nodelist[_iml];
	NODED(_nd) += _g;
  }
 
}
 
}

static void nrn_state(_NrnThread* _nt, _Memb_list* _ml, int _type) {

}

static void terminal(){}

static void _initlists(){
 double _x; double* _p = &_x;
 int _i; static int _first = 1;
  if (!_first) return;
_first = 0;
}

#if defined(__cplusplus)
} /* extern "C" */
#endif

#if NMODL_TEXT
static const char* nmodl_filename = "/home/meowlin/projects/LFP_kernel_files/LFPykernels-main/examples/mod/Ih_linearized_v2_frozen.mod";
static const char* nmodl_file_text = 
  ":Comment : Linearized by Torbjorn Ness 2013\n"
  ":Reference : :		Kole,Hallermann,and Stuart, J. Neurosci. 2006\n"
  "\n"
  "NEURON	{\n"
  "	SUFFIX Ih_linearized_v2_frozen\n"
  "	NONSPECIFIC_CURRENT ihcn\n"
  "	RANGE gIhbar, V_R, ihcn, wInf, ehcn, phi, wTau\n"
  "}\n"
  "\n"
  "UNITS	{\n"
  "	(S) = (siemens)\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "}\n"
  "\n"
  "PARAMETER	{\n"
  "	gIhbar = 0.00001	(S/cm2) \n"
  "	ehcn =  -45.0 		(mV)\n"
  "	V_R		(mV)\n"
  "}\n"
  "\n"
  "ASSIGNED	{\n"
  "	v	(mV)\n"
  "	ihcn	(mA/cm2)\n"
  "	wInf\n"
  "	wTau\n"
  "	wAlpha\n"
  "	a1\n"
  "	a2\n"
  "	a3\n"
  "	b1\n"
  "	b2\n"
  "	b3\n"
  "	foo\n"
  "	dwinf\n"
  "	wBeta\n"
  "    phi\n"
  "}\n"
  "\n"
  "BREAKPOINT	{\n"
  "	:SOLVE states METHOD cnexp\n"
  "	ihcn  = gIhbar*wInf*(v-ehcn) - wInf*phi\n"
  "}\n"
  "\n"
  "\n"
  "INITIAL{\n"
  "	a1	= 0.001*6.43\n"
  "	a2 	= 154.9\n"
  "	a3 	= 11.9\n"
  "	b1 	= 0.001*193\n"
  "	b2 	= 33.1\n"
  "	wAlpha 	= a1*(V_R+a2)/(exp((V_R+a2)/a3)-1)\n"
  "	wBeta  	=  b1*exp(V_R/b2)\n"
  "	wInf 	= wAlpha/(wAlpha + wBeta)\n"
  "	foo 	= wAlpha/(a3*a1) * exp((V_R + a2)/a3) + (V_R + a2)/b2 - 1\n"
  "    wTau 	= 1/(wAlpha + wBeta)\n"
  "    dwinf 	= - wBeta*wAlpha *wTau * wTau /(V_R + a2) * foo\n"
  "    phi = gIhbar * dwinf *(ehcn - V_R)\n"
  "}\n"
  ;
#endif
