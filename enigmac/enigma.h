#ifndef ENIGMA_H_
#define ENIGMA_H_

#include <stddef.h>

#define MAX_LENGTH 2
#define MAXTURNOVERNUM (5)

typedef struct wheel__ {
    char wheel[MAX_LENGTH + 1];
} Wheel;

typedef struct wheelpos__ {
    char wheelpos;
} WheelPos;

typedef struct wheelring__ {
    char wheelring;
} WheelRing;

typedef struct turnover__ {
    char turnover[MAXTURNOVERNUM + 1];
} Turnover;

typedef struct param__ {
    Wheel wheel[6];
    WheelPos wheelpos[4];
    WheelRing wheelring[4];
    Wheel plugboard;
    Turnover turnover[3];
    int reflectorsetflag;
    int thinflag;
} EnigmaParam;

typedef struct cursetting__ {
    int indicator[4];
    int prevturnoverflag[3];
} CurSetting;

typedef struct enigma_settings__ {
    int reflector;
    int wheel_order[3];
    char ring_setting[3];
    char wheel_pos[3];
    char* plugboard_pairs;
    const unsigned int plaintextsize;
    char* from;
} EnigmaSettings;

typedef enum rotorenum__ {
    ROTOR_I, ROTOR_II, ROTOR_III, ROTOR_IV, ROTOR_V,
} RotorEnum;

static const Wheel RotorSet[] = {
    {"EKMFLGDQVZNTOWYHXUSPAIBRCJ"},
    {"AJDKSIRUXBLHWTMCQGZNPYFVOE"},
    {"BDFHJLCPRTXVZNYEIWGAKMUSQO"},
    {"ESOVPZJAYQUIRHXLNFTGKDCMWB"},
    {"VZBRGITYUPSDNHLXAWMJQOFECK"},
};

typedef enum reflectorenum__ {
    REFLECTOR_A, REFLECTOR_B, REFLECTOR_C,
} ReflectorEnum;

static const Wheel ReflectorSet[] = {
    {"EJMZALYXVBWFCRQUONTSPIKHGD"},
    {"YRUHQSLDPXNGOKMIEBFZCWVJAT"},
    {"FVPJIAOYEDRZXWGCTKUQSBNMHL"},
};

typedef enum etwenum__ {
    ETW,
    ETW_KEYBOARD,
    ETW_T,
} ETWEnum;

static const Wheel ETWSet[] = {
    {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"},
};

typedef enum turnoverenum__ {
    TURNOVER_I, TURNOVER_II, TURNOVER_III, TURNOVER_IV, TURNOVER_V,
} TurnoverEnum;

static const Turnover TurnoverSet[] = {
    {"QQQQQ"}, {"EEEEE"}, {"VVVVV"}, {"JJJJJ"}, {"ZZZZZ"},
};

static const Wheel IDENTITY_WHEEL = {"ABCDEFGHIJKLMNOPQRSTUVWXYZ"};

int setplugboard_from_plugboardpairs(EnigmaParam* param, const char* plugboard_pairs);

int setparam_ENIGMA_I(
    EnigmaParam* param,
    int reflector, int wheel_order[3],
    const char ring_setting[3], const char wheel_pos[3],
    const char* plugboard_pairs
);

int resetCurrentSetting(CurSetting* cursetting, EnigmaParam* param);

int saveCurrentSetting(EnigmaParam* param, CurSetting* cursetting);

int setInverseWheel(EnigmaParam* param, Wheel inversewheel[]);

int incrementIndicator_normal(CurSetting* cursetting, EnigmaParam* param);

char wheelout(const char wheel[26], int offset, char ch);

char enigmaEachEncrypt(
    EnigmaParam* param, CurSetting* cursetting,
    Wheel inversewheel[],
    char ch
);

const char* run_enigma(int reflector, int wheel_order[], char ring_setting[], char wheel_pos[], char plugboard_pairs[], const unsigned int plaintextsize, char from[] );

#endif