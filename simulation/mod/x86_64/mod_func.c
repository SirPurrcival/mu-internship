#include <stdio.h>
#include "hocdec.h"
extern int nrnmpi_myid;
extern int nrn_nobanner_;

extern void _exp2synI_reg(void);
extern void _Ih_linearized_v2_frozen_reg(void);
extern void _NaTa_t_frozen_reg(void);
extern void _SKv3_1_frozen_reg(void);

void modl_reg(){
  if (!nrn_nobanner_) if (nrnmpi_myid < 1) {
    fprintf(stderr, "Additional mechanisms from files\n");

    fprintf(stderr," \"exp2synI.mod\"");
    fprintf(stderr," \"Ih_linearized_v2_frozen.mod\"");
    fprintf(stderr," \"NaTa_t_frozen.mod\"");
    fprintf(stderr," \"SKv3_1_frozen.mod\"");
    fprintf(stderr, "\n");
  }
  _exp2synI_reg();
  _Ih_linearized_v2_frozen_reg();
  _NaTa_t_frozen_reg();
  _SKv3_1_frozen_reg();
}
