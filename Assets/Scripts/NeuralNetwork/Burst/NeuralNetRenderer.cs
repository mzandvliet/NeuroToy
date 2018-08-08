using UnityEngine;
using System.Collections.Generic;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Collections;
using Unity.Mathematics;

/*
    - Approach 1: have two matrix arrays. Matrix4x4, float4x4.
    Make a map function, either using slow loop copy, or straight memcopy
    Use DrawMeshInstanced to send cpu-side object params each frame
    - Approach 2: rendering a compute buffer of objects
    Update that with data from NativeArray
    - Approach 3:
    For memory that is already the result of gpu compute, just handle it
    on there entirely.

    Which do I need first? Well, I am working on the cpu-side of neural
    compute first, and I definitely need to visualize that state.

    ------

    cpu-side idea with memcopy first.
    Ok, this is very useful already. :P
    One downside: 1023 instanced per DrawMeshInstanced call.
    Another downside: While this is faster than I'm used to, the renderer
    probably still takes the managed state and marshalls it back to
    unmanaged land. Unity will have to create the fastest path themselves.

     */

namespace NNBurst {
    public class NeuralNetRenderer : MonoBehaviour {
        [SerializeField] private Mesh _mesh;
        [SerializeField] private Material _mat;

        private Matrix4x4[] _objectsRender;
        private NativeArray<float4x4> _objectsNative;


        private void Awake() {
            const int count = 512;
            _objectsRender = new Matrix4x4[count];
            _objectsNative = new NativeArray<float4x4>(count, Allocator.Persistent, NativeArrayOptions.ClearMemory);

            for (int i = 0; i < count; i++) {
                float phase = i / 512f * Mathf.PI * 2f * 10f;
                float s, c;
                math.sincos(phase, out s, out c); // Oh!
                var mat = new float4x4(quaternion.identity, new float3(c * 10f, i / 64f, s * 10f));
                
                _objectsNative[i] = mat;
            }
        }

        private void OnDestroy() {
            _objectsNative.Dispose();
        }

        unsafe static void GetNativeMatrix4x4Array(NativeArray<float4x4> from, Matrix4x4[] to) {
            fixed(void* toPointer = to) {
                void* fromPointer = NativeArrayUnsafeUtility.GetUnsafeBufferPointerWithoutChecks(from);
                UnsafeUtility.MemCpy(toPointer, fromPointer, from.Length * (long)UnsafeUtility.SizeOf<float4x4>());
            }
        }

        private void Update() {
            // Todo: start jobs that transform neural net state to render state            
        }

        private void LateUpdate() {
            GetNativeMatrix4x4Array(_objectsNative, _objectsRender);
            Graphics.DrawMeshInstanced(_mesh, 0, _mat, _objectsRender);
            // Todo: probably need material property blocks for things like per-object color
        }
    }
}
