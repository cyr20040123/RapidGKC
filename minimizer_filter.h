// #define FILTER_KERNEL new_filter2 // modify this to change filter: mm_filter, sign_filter, new_filter, new_filter2
// #define CANO_FILTER
// #define SIGN_FILTER

#ifdef CANO_FILTER
    #define MM_FILTER return true;
    #define FILTER_KERNEL canonical
#else
    #ifdef SIGN_FILTER
    #define FILTER_KERNEL signature
    #define MM_FILTER T_minimizer t = mm;\
        bool flag = true;\
        for (int ii = 0; ii < p-2; ii ++) {\
            flag *= ((t & 0b1111) != 0);\
            t = t >> 2;\
        }\
        return flag & (((mm >> ((p-3)*2)) & 0b111011) != 0); /*AAA ACA*/;
    #else
        #define FILTER_KERNEL rapidgkc
        #define MM_FILTER return ((((mm >> ((p-3)*2)) & 0b101011) != 0/*no AAA ACA CAA CCA*/) & ((mm & 0b111111) != 0/*no AAA at last*/));
    #endif
#endif
