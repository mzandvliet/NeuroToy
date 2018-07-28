using UnityEngine;
using Unity.Collections;
using Unity.Jobs;
using DataManager = NNBurst.Mnist.DataManager;
using NNBurst;

public class ConvTest : MonoBehaviour {
    private int _imgLabel;
    private Texture2D _imgTex;
    private Texture2D _actTex;

    private System.Random _random;
    
    
    private void Awake() {
        DataManager.Load();
        _random = new System.Random();

        const int kSize = 3;
        const int stride = 1;

        var kernel = new NativeArray<float>(kSize*kSize, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var img = new NativeArray<float>(28 * 28, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);
        var act = new NativeArray<float>(26 * 26, Allocator.Persistent, NativeArrayOptions.UninitializedMemory);

        const int imgIdx = 0;
        _imgLabel = DataManager.Train.Labels[imgIdx];
        var handle = NeuralJobs.CopyInput(img, DataManager.Train, imgIdx);

        NeuralMath.RandomGaussian(_random, kernel, 0f, 1f);

        var cj = new Conv2DJob();
        cj.input = img;
        cj.output = act;
        cj.kernel = kernel;
        cj.stride = stride;
        handle = cj.Schedule(handle);

        handle.Complete();

        _imgTex = new Texture2D(28, 28, TextureFormat.ARGB32, false, true);
        _actTex = new Texture2D(26, 26, TextureFormat.ARGB32, false, true);
        ToTexture(img, _imgTex);
        ToTexture(act, _actTex);

        kernel.Dispose();
        img.Dispose();
        act.Dispose();
    }

    private static void ToTexture(NativeArray<float> img, Texture2D tex) {
        var colors = new Color[img.Length];

        for (int y = 0; y < tex.height; y++) {
            for (int x = 0; x < tex.width; x++) {
                float pix = img[y * tex.height + x];
                // Invert y
                colors[(tex.height - 1 - y) * tex.height + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, tex.width, tex.height, colors);
        tex.Apply(false);
    }

    private void OnGUI() {
        GUI.Label(new Rect(0f, 32f, 280f, 32f), "Label: " + _imgLabel);
        GUI.DrawTexture(new Rect(0f, 64f, 280f, 280f), _imgTex, ScaleMode.ScaleToFit);
        GUI.DrawTexture(new Rect(10f, 64f + 280f, 260f, 260f), _actTex, ScaleMode.ScaleToFit);
    }
}