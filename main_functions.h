

#ifndef _MAIN_FUNCTIONS_H_
#define _MAIN_FUNCTIONS_H_

#include "esp_camera.h"

// Expose a C friendly interface for main functions.
#ifdef __cplusplus
extern "C" {
#endif

void setup(void);

void loop(uint8_t *buf, size_t len);

#ifdef __cplusplus
}
#endif

#endif
