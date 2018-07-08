// Adapted from: https://github.com/frozax/misc/blob/master/fgRandom/fgRandom.cs

/***************************
 * fgRandom.cs
 *
 * C# Mersenne Twister implementation
 *
 * Based on https://en.wikipedia.org/wiki/Mersenne_Twister
 *
 * (c) Francois GUIBERT
 * www.frozax.com
 *
 * Feel free to use in any commercial or personal projects
 *
 * Send me a tweet if you use or like it: @Frozax
 *
 * **********************************/

using System;
using Unity.Collections;

public struct MTRandom {
    private const int N = 624;
    private const int M = 397;
    private int _index;
    private NativeArray<uint> _mt;

    public MTRandom(uint seed) {
        _index = N;
        _mt = new NativeArray<uint>(N, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        _mt[0] = seed;
        for (int i = 1; i < N; i++) {
            _mt[i] = 1812433253 * (_mt[i - 1] ^ (_mt[i - 1] >> 30)) + (uint)i;
        }
    }

    private uint ExtractNumber() {
        if (_index >= N)
            Twist();

        uint y = _mt[_index];

        // right shift by 11 bits
        y = y ^ (y >> 11);
        y = y ^ ((y << 7) & 2636928640);
        y = y ^ ((y << 15) & 4022730752);
        y = y ^ (y >> 18);
        _index++;

        return y;
    }

    private uint _int32(long x) {
        return (uint)(0xFFFFFFF & x);
    }

    private void Twist() {
        for (int i = 0; i < N; i++) {
            uint y = ((_mt[i]) & 0x80000000) +
                ((_mt[(i + 1) % N]) & 0x7fffffff);
            _mt[i] = _mt[(i + M) % N] ^ (uint)(y >> 1);
            if (y % 2 != 0)
                _mt[i] = _mt[i] ^ 0x9908b0df;
        }
        _index = 0;
    }

    // Real functions
    public uint NextUInt() { return ExtractNumber(); }
    // Can be negative
    public int NextInt() { return unchecked((int)ExtractNumber()); }
    // max is NOT included
    public int NextInt(int max) { return (int)(NextUInt() % max); }
    // between min (included) and max (excluded)
    public int NextInt(int min, int max) { return (int)(NextUInt() % (max - min) + min); }
    // between 0 and 1 (included)
    public float NextFloat() { return (float)(NextUInt() % 65536) / 65535.0f; }
}