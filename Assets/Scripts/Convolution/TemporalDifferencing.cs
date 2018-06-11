using System.Runtime.InteropServices;
using UnityEngine;

/* Todo:
 * - Basic camera-compute-visualize pipeline. Suggest turning to grayscale.
 */

/*public struct Pixel {
    public float Value;
}*/

public class TemporalDifferencing : MonoBehaviour {
    [SerializeField] private ComputeShader _compute;
    [SerializeField] private int _res = 128;

    private Camera _camera;
    
    private RenderTexture _retinaTex;
    private RenderTexture _outputTex;

    ComputeBuffer _buff;

    private int _grayscaleKernel;
    private int _grayscaleKernelLength = 32;
    

    void Awake() {
        Application.targetFrameRate = 60;

        _retinaTex = new RenderTexture(_res, _res, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        _retinaTex.enableRandomWrite = true;
        _retinaTex.Create();

        _outputTex = new RenderTexture(_res, _res, 0, RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        _outputTex.enableRandomWrite = true;
        _outputTex.Create();

        _camera = gameObject.GetComponent<Camera>();
        _camera.targetTexture = _retinaTex;
        _camera.enabled = false;

        _buff = new ComputeBuffer(_res * _res, Marshal.SizeOf(typeof(float)) * 3);
        
        _grayscaleKernel = _compute.FindKernel("CSToGray");

        _compute.SetInt("Res", _res);
        _compute.SetTexture(_grayscaleKernel, "RetinaTex", _retinaTex);
        _compute.SetTexture(_grayscaleKernel, "OutputTex", _outputTex);
        _compute.SetBuffer(_grayscaleKernel, "Buff", _buff);
    }

    private void OnDestroy() {
        _retinaTex.Release();
        _outputTex.Release();

        _buff.Release();
    }

    private void Update() {
        _camera.Render();
        _compute.Dispatch(_grayscaleKernel, _res / _grayscaleKernelLength, _res / _grayscaleKernelLength, 1);
    }

    private void OnGUI() {
        GUI.DrawTexture(new Rect(0,0, Screen.height, Screen.height), _outputTex, ScaleMode.ScaleAndCrop);
    }
}
