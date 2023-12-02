#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "enigma.h"

int setplugboard_from_plugboardpairs(EnigmaParam* param, const char* plugboard_pairs) {
    unsigned int i, j;
    size_t len = strnlen(plugboard_pairs, 26 + 1);

    for (i = 0; i < len; i++) {
        if ((plugboard_pairs[i] < 'A') || (plugboard_pairs[i] > 'Z'))
            return -1;
        for (j = i + 1; j < len; j++) {
            if (plugboard_pairs[i] == plugboard_pairs[j])
                return -1;
        }
    }

    for (i = 0; i < 26; i++)
        param->plugboard.wheel[i] = (char)('A' + i);

    for (i = 0; i < len; i += 2) {
        param->plugboard.wheel[plugboard_pairs[i] - 'A'] = plugboard_pairs[i + 1];
        param->plugboard.wheel[plugboard_pairs[i + 1] - 'A'] = plugboard_pairs[i];
    }

    return 0;
}

int setparam_ENIGMA_I(
    EnigmaParam* param,
    int reflector, int wheel_order[3],
    const char ring_settting[3], const char wheel_pos[3],
    const char* plugboard_pairs
) {
    unsigned int i;

    param->wheel[0] = ETWSet[ETW];
    param->wheel[5] = IDENTITY_WHEEL;  

    for (i=0; i < 3; i++) {
        switch (wheel_order[i]) {
            case 0:
                param->wheel[(3-i)] = RotorSet[ROTOR_I];
                param->turnover[(2-i)] = TurnoverSet[TURNOVER_I];
                break;
            case 1:
                param->wheel[(3-i)] = RotorSet[ROTOR_II];
                param->turnover[(2-i)] = TurnoverSet[TURNOVER_II];
                break;
            case 2:
                param->wheel[(3-i)] = RotorSet[ROTOR_III];
                param->turnover[(2-i)] = TurnoverSet[TURNOVER_III];
                break;
            case 3:
                param->wheel[(3-i)] = RotorSet[ROTOR_IV];
                param->turnover[(2-i)] = TurnoverSet[TURNOVER_IV];
                break;
            case 4:
                param->wheel[(3-i)] = RotorSet[ROTOR_V];
                param->turnover[(2-i)] = TurnoverSet[TURNOVER_V];
                break;
            default:
                return -1;  // invalid wheel_order
        }
    }

    switch (reflector) {
        case 0:
            param->wheel[4] = ReflectorSet[REFLECTOR_A];
            break;
        case 1:
            param->wheel[4] = ReflectorSet[REFLECTOR_B];
            break;
        case 2:
            param->wheel[4] = ReflectorSet[REFLECTOR_C];
            break;
        default:
            return -1;
    }

    for (i=0; i < 3; i++) {
        if ((wheel_pos[i] >= 'A') && (wheel_pos[i] <= 'Z')) {
            param->wheelpos[(2-i)].wheelpos = wheel_pos[i];
        } else {
            return -1;
        }
    }
    param->wheelpos[3].wheelpos = 'A';

    for (i=0; i < 3; i++) {
        if ((ring_settting[i] >= 'A') && (ring_settting[i] <= 'Z')) {
            param->wheelring[(2-i)].wheelring = ring_settting[i];
        } else {
            return -1;
        }
    }
    param->wheelring[3].wheelring = 'A';

    if (setplugboard_from_plugboardpairs(param, plugboard_pairs) != 0)
        return -1;

    param->reflectorsetflag = 0;
    param->thinflag = 0;

    return 0;
}


int resetCurrentSetting(CurSetting* cursetting, EnigmaParam* param) {
    unsigned int i, j;

    for (i=0; i < 4; i++) {
        cursetting->indicator[i] = param->wheelpos[i].wheelpos - 'A';

        if (i != 3) {
            cursetting->prevturnoverflag[i] = 0;
            for (j=0; j < MAXTURNOVERNUM; j++) {
                cursetting->prevturnoverflag[i] =
                    cursetting->prevturnoverflag[i] ||
                    (cursetting->indicator[i] + 'A' ==
                        param->turnover[i].turnover[j]);
            }
        }
    }

    return 0;
}

int saveCurrentSetting(EnigmaParam* param, CurSetting* cursetting) {
    unsigned int i;

    for (i=0; i < 4; i++) {
        param->wheelpos[i].wheelpos = cursetting->indicator[i] + 'A';
    }

    return 0;
}

int setInverseWheel(EnigmaParam* param, Wheel inversewheel[]) {
    unsigned int i, j;

    for (i=0; i < 6; i++) {
        for (j=0; j < 26; j++)
            inversewheel[i].wheel[param->wheel[i].wheel[j] - 'A'] = j + 'A';
    }

    return 0;
}


int incrementIndicator_normal(CurSetting* cursetting, EnigmaParam* param) {
    unsigned int i, j;

    for (i=0; i < 2; i++) {
        if (cursetting->prevturnoverflag[i] == 1) {
            if ((i != 0) && (cursetting->prevturnoverflag[i-1] == 0)) {
                cursetting->indicator[i] += 1;
                if (cursetting->indicator[i] >= 26)
                    cursetting->indicator[i] -= 26;
            }
            cursetting->indicator[i+1] += 1;
            if (cursetting->indicator[i+1] >= 26)
                cursetting->indicator[i+1] -= 26;
        }
    }
    cursetting->indicator[0] += 1;
    if (cursetting->indicator[0] >= 26)
        cursetting->indicator[0] -= 26;


    for (i=0; i < 3; i++) {
        cursetting->prevturnoverflag[i] = 0;
        for (j=0; j < MAXTURNOVERNUM; j++)
            cursetting->prevturnoverflag[i] =
                cursetting->prevturnoverflag[i] ||
                (cursetting->indicator[i] + 'A' ==
                    param->turnover[i].turnover[j]);
    }

    return 0;
}


char wheelout(const char wheel[26], int offset, char ch) {
    int preidx = (ch - 'A') + offset;

    if (preidx < 0)
        preidx += 26;
    if (preidx >= 26)
        preidx -= 26;

    int postidx = (wheel[preidx] - 'A') - offset;
    if (postidx < 0)
        postidx += 26;
    if (postidx >= 26)
        postidx -= 26;

    return postidx + 'A';
}

char enigmaEachEncrypt(
    EnigmaParam* param, CurSetting* cursetting,
    Wheel inversewheel[],
    char ch
) {
    incrementIndicator_normal(cursetting, param);

    ch = wheelout(param->plugboard.wheel, 0, ch);

    ch = inversewheel[0].wheel[ch - 'A'];  

    ch = wheelout(
        param->wheel[1].wheel,
        cursetting->indicator[0] - (param->wheelring[0].wheelring - 'A'),
        ch);

    ch = wheelout(
        param->wheel[2].wheel,
        cursetting->indicator[1] - (param->wheelring[1].wheelring - 'A'),
        ch);

    ch = wheelout(
        param->wheel[3].wheel,
        cursetting->indicator[2] - (param->wheelring[2].wheelring - 'A'),
        ch);

    ch = wheelout(
        param->wheel[4].wheel,
        cursetting->indicator[3] - (param->wheelring[3].wheelring - 'A'),
        ch);

    if (param->thinflag == 1) {
        ch = wheelout(param->wheel[5].wheel, 0, ch);

        ch = wheelout(
            inversewheel[4].wheel,
            cursetting->indicator[3] - (param->wheelring[3].wheelring - 'A'),
            ch);
    }

    ch = wheelout(
        inversewheel[3].wheel,
        cursetting->indicator[2] - (param->wheelring[2].wheelring - 'A'),
        ch);

    ch = wheelout(
        inversewheel[2].wheel,
        cursetting->indicator[1] - (param->wheelring[1].wheelring - 'A'),
        ch);

    ch = wheelout(
        inversewheel[1].wheel,
        cursetting->indicator[0] - (param->wheelring[0].wheelring - 'A'),
        ch);

    ch = param->wheel[0].wheel[ch - 'A'];  

    ch = wheelout(param->plugboard.wheel, 0, ch);

    return ch;
}

const char* run_enigma(int reflector, int wheel_order[], char ring_setting[], char wheel_pos[], char plugboard_pairs[], const unsigned int plaintextsize, char from[] ) {
    EnigmaParam param;
    char to[plaintextsize + 1];
    setparam_ENIGMA_I(&param, reflector, wheel_order, ring_setting, wheel_pos, (char*)plugboard_pairs);

    CurSetting cursetting;
    Wheel inversewheel[6];
    unsigned int len = strlen(from);

    resetCurrentSetting(&cursetting, &param);
    setInverseWheel(&param, inversewheel);

    for (unsigned int i=0; i < len; i++)
        to[i] = enigmaEachEncrypt(&param, &cursetting, inversewheel, from[i]);

    to[len] = '\0';
    return strdup(to);
}
