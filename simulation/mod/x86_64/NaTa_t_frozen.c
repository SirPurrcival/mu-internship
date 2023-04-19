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
 
#define nrn_init _nrn_init__NaTa_t_frozen
#define _nrn_initial _nrn_initial__NaTa_t_frozen
#define nrn_cur _nrn_cur__NaTa_t_frozen
#define _nrn_current _nrn_current__NaTa_t_frozen
#define nrn_jacob _nrn_jacob__NaTa_t_frozen
#define nrn_state _nrn_state__NaTa_t_frozen
#define _net_receive _net_receive__NaTa_t_frozen 
 
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
#define gNaTa_tbar _p[0]
#define V_R _p[1]
#define ina _p[2]
#define gNaTa_t _p[3]
#define ena _p[4]
#define mInf _p[5]
#define mTau _p[6]
#define mAlpha _p[7]
#define mBeta _p[8]
#define hInf _p[9]
#define hTau _p[10]
#define hAlpha _p[11]
#define hBeta _p[12]
#define m _p[13]
#define h _p[14]
#define v _p[15]
#define _g _p[16]
#define _ion_ena	*_ppvar[0]._pval
#define _ion_ina	*_ppvar[1]._pval
#define _ion_dinadv	*_ppvar[2]._pval
 
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
 "setdata_NaTa_t_frozen", _hoc_setdata,
 0, 0
};
 /* declare global and static user variables */
 /* some parameters have upper and lower limits */
 static HocParmLimits _hoc_parm_limits[] = {
 0,0,0
};
 static HocParmUnits _hoc_parm_units[] = {
 "gNaTa_tbar_NaTa_t_frozen", "S/cm2",
 "V_R_NaTa_t_frozen", "mV",
 "ina_NaTa_t_frozen", "mA/cm2",
 "gNaTa_t_NaTa_t_frozen", "S/cm2",
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
"NaTa_t_frozen",
 "gNaTa_tbar_NaTa_t_frozen",
 "V_R_NaTa_t_frozen",
 0,
 "ina_NaTa_t_frozen",
 "gNaTa_t_NaTa_t_frozen",
 0,
 0,
 0};
 static Symbol* _na_sym;
 
extern Prop* need_memb(Symbol*);

static void nrn_alloc(Prop* _prop) {
	Prop *prop_ion;
	double *_p; Datum *_ppvar;
 	_p = nrn_prop_data_alloc(_mechtype, 17, _prop);
 	/*initialize range parameters*/
 	gNaTa_tbar = 1e-05;
 	V_R = 0;
 	_prop->param = _p;
 	_prop->param_size = 17;
 	_ppvar = nrn_prop_datum_alloc(_mechtype, 3, _prop);
 	_prop->dparam = _ppvar;
 	/*connect ionic variables to this model*/
 prop_ion = need_memb(_na_sym);
 nrn_promote(prop_ion, 0, 1);
 	_ppvar[0]._pval = &prop_ion->param[0]; /* ena */
 	_ppvar[1]._pval = &prop_ion->param[3]; /* ina */
 	_ppvar[2]._pval = &prop_ion->param[4]; /* _ion_dinadv */
 
}
 static void _initlists();
 static void _update_ion_pointer(Datum*);
 extern Symbol* hoc_lookup(const char*);
extern void _nrn_thread_reg(int, int, void(*)(Datum*));
extern void _nrn_thread_table_reg(int, void(*)(double*, Datum*, Datum*, _NrnThread*, int));
extern void hoc_register_tolerance(int, HocStateTolerance*, Symbol***);
extern void _cvode_abstol( Symbol**, double*, int);

 void _NaTa_t_frozen_reg() {
	int _vectorized = 1;
  _initlists();
 	ion_reg("na", -10000.);
 	_na_sym = hoc_lookup("na_ion");
 	register_mech(_mechanism, nrn_alloc,nrn_cur, nrn_jacob, nrn_state, nrn_init, hoc_nrnpointerindex, 1);
 _mechtype = nrn_get_mechtype(_mechanism[1]);
     _nrn_setdata_reg(_mechtype, _setdata);
     _nrn_thread_reg(_mechtype, 2, _update_ion_pointer);
 #if NMODL_TEXT
  hoc_reg_nmodl_text(_mechtype, nmodl_file_text);
  hoc_reg_nmodl_filename(_mechtype, nmodl_filename);
#endif
  hoc_register_prop_size(_mechtype, 17, 3);
  hoc_register_dparam_semantics(_mechtype, 0, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 1, "na_ion");
  hoc_register_dparam_semantics(_mechtype, 2, "na_ion");
 	hoc_register_var(hoc_scdoub, hoc_vdoub, hoc_intfunc);
 	ivoc_help("help ?1 NaTa_t_frozen /home/meowlin/projects/LFP_kernel_files/LFPykernels-main/examples/mod/NaTa_t_frozen.mod\n");
 hoc_register_limits(_mechtype, _hoc_parm_limits);
 hoc_register_units(_mechtype, _hoc_parm_units);
 }
static int _reset;
static char *modelname = "";

static int error;
static int _ninits = 0;
static int _match_recurse=1;
static void _modl_cleanup(){ _match_recurse=1;}
 extern void nrn_update_ion_pointer(Symbol*, Datum*, int, int);
 static void _update_ion_pointer(Datum* _ppvar) {
   nrn_update_ion_pointer(_na_sym, _ppvar, 0, 0);
   nrn_update_ion_pointer(_na_sym, _ppvar, 1, 3);
   nrn_update_ion_pointer(_na_sym, _ppvar, 2, 4);
 }

static void initmodel(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt) {
  int _i; double _save;{
 {
   double _lqt ;
 _lqt = pow( 2.3 , ( ( 34.0 - 21.0 ) / 10.0 ) ) ;
    if ( V_R  == - 38.0 ) {
     V_R = V_R + 0.0001 ;
     }
   mAlpha = ( 0.182 * ( V_R - - 38.0 ) ) / ( 1.0 - ( exp ( - ( V_R - - 38.0 ) / 6.0 ) ) ) ;
   mBeta = ( 0.124 * ( - V_R - 38.0 ) ) / ( 1.0 - ( exp ( - ( - V_R - 38.0 ) / 6.0 ) ) ) ;
   mTau = ( 1.0 / ( mAlpha + mBeta ) ) / _lqt ;
   mInf = mAlpha / ( mAlpha + mBeta ) ;
   if ( V_R  == - 66.0 ) {
     V_R = V_R + 0.0001 ;
     }
   hAlpha = ( - 0.015 * ( V_R - - 66.0 ) ) / ( 1.0 - ( exp ( ( V_R - - 66.0 ) / 6.0 ) ) ) ;
   hBeta = ( - 0.015 * ( - V_R - 66.0 ) ) / ( 1.0 - ( exp ( ( - V_R - 66.0 ) / 6.0 ) ) ) ;
   hTau = ( 1.0 / ( hAlpha + hBeta ) ) / _lqt ;
   hInf = hAlpha / ( hAlpha + hBeta ) ;
    m = mInf ;
   h = hInf ;
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
  ena = _ion_ena;
 initmodel(_p, _ppvar, _thread, _nt);
 }
}

static double _nrn_current(double* _p, Datum* _ppvar, Datum* _thread, _NrnThread* _nt, double _v){double _current=0.;v=_v;{ {
   gNaTa_t = gNaTa_tbar * m * m * m * h ;
   ina = gNaTa_t * ( v - ena ) ;
   }
 _current += ina;

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
  ena = _ion_ena;
 _g = _nrn_current(_p, _ppvar, _thread, _nt, _v + .001);
 	{ double _dina;
  _dina = ina;
 _rhs = _nrn_current(_p, _ppvar, _thread, _nt, _v);
  _ion_dinadv += (_dina - ina)/.001 ;
 	}
 _g = (_g - _rhs)/.001;
  _ion_ina += ina ;
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
static const char* nmodl_filename = "/home/meowlin/projects/LFP_kernel_files/LFPykernels-main/examples/mod/NaTa_t_frozen.mod";
static const char* nmodl_file_text = 
  ":Reference :Colbert and Pan 2002\n"
  "\n"
  "NEURON	{\n"
  "	SUFFIX NaTa_t_frozen\n"
  "	USEION na READ ena WRITE ina\n"
  "	RANGE gNaTa_tbar, gNaTa_t, ina, V_R\n"
  "}\n"
  "\n"
  "UNITS	{\n"
  "	(S) = (siemens)\n"
  "	(mV) = (millivolt)\n"
  "	(mA) = (milliamp)\n"
  "}\n"
  "\n"
  "PARAMETER	{\n"
  "	gNaTa_tbar = 0.00001 (S/cm2)\n"
  "	V_R (mV)\n"
  "}\n"
  "\n"
  "ASSIGNED	{\n"
  "	v	(mV)\n"
  "	ena	(mV)\n"
  "	ina	(mA/cm2)\n"
  "	gNaTa_t	(S/cm2)\n"
  "	mInf\n"
  "	mTau\n"
  "	mAlpha\n"
  "	mBeta\n"
  "	hInf\n"
  "	hTau\n"
  "	hAlpha\n"
  "	hBeta\n"
  "    m\n"
  "    h\n"
  "}\n"
  "\n"
  "BREAKPOINT	{\n"
  "	gNaTa_t = gNaTa_tbar*m*m*m*h\n"
  "	ina = gNaTa_t*(v-ena)\n"
  "}\n"
  "\n"
  "INITIAL{\n"
  "  LOCAL qt\n"
  "  qt = 2.3^((34-21)/10)\n"
  "\n"
  "  UNITSOFF\n"
  "    if(V_R == -38){\n"
  "    	V_R = V_R+0.0001\n"
  "    }\n"
  "		mAlpha = (0.182 * (V_R- -38))/(1-(exp(-(V_R- -38)/6)))\n"
  "		mBeta  = (0.124 * (-V_R -38))/(1-(exp(-(-V_R -38)/6)))\n"
  "		mTau = (1/(mAlpha + mBeta))/qt\n"
  "		mInf = mAlpha/(mAlpha + mBeta)\n"
  "\n"
  "    if(V_R == -66){\n"
  "      V_R = V_R + 0.0001\n"
  "    }\n"
  "\n"
  "		hAlpha = (-0.015 * (V_R- -66))/(1-(exp((V_R- -66)/6)))\n"
  "		hBeta  = (-0.015 * (-V_R -66))/(1-(exp((-V_R -66)/6)))\n"
  "		hTau = (1/(hAlpha + hBeta))/qt\n"
  "		hInf = hAlpha/(hAlpha + hBeta)\n"
  "	UNITSON\n"
  "	m = mInf\n"
  "	h = hInf\n"
  "}\n"
  ;
#endif
