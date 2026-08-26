#pragma once
// test_framework.h — Minimal test framework shared across test files
//
// Provides EXPECT_TRUE, EXPECT_FALSE, EXPECT_EQ, EXPECT_NEAR macros
// and global pass/fail counters. Include once per test .cpp (before
// any test code).

#include <cmath>
#include <cstdio>

static int g_pass = 0;
static int g_fail = 0;

#define EXPECT_TRUE(expr)                                                      \
    do {                                                                       \
        if (expr) {                                                            \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: expected true: %s\n", __FILE__,    \
                    __LINE__, #expr);                                          \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT_FALSE(expr) EXPECT_TRUE(!(expr))

#define EXPECT_EQ(a, b) EXPECT_TRUE((a) == (b))

#define EXPECT_NEAR(a, b, eps)                                                 \
    do {                                                                       \
        double _a = (double)(a), _b = (double)(b), _e = (double)(eps);        \
        double _diff = fabs(_a - _b);                                          \
        if (_diff <= _e) {                                                     \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: |%s - %s| = %g > %g\n",           \
                    __FILE__, __LINE__, #a, #b, _diff, _e);                   \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

#define EXPECT_STR_EQ(a, b)                                                    \
    do {                                                                       \
        std::string _a = (a), _b = (b);                                       \
        if (_a == _b) {                                                        \
            ++g_pass;                                                          \
        } else {                                                               \
            fprintf(stderr, "FAIL [%s:%d]: \"%s\" != \"%s\"\n",              \
                    __FILE__, __LINE__, _a.c_str(), _b.c_str());              \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)
