using System.IO;
using System.Text;
using UnityEngine;

/* Todo: load test and validation data, not just training. */

public class Mnist {
    private const string Folder = "F:\\datasets\\mnist";
    private const string ImagePath = "train-images.idx3-ubyte";
    private const string LabelPath = "train-labels.idx1-ubyte";

    public static int NumImgs = -1;
    public static int Rows = -1;
    public static int Cols = -1;
    public static int ImgDims = -1;

    public static void Load(out float[] pixels, out int[] labels) {
        var lblReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, LabelPath), FileMode.Open));
        var imgReader = new BigEndianBinaryReader(new FileStream(Path.Combine(Folder, ImagePath), FileMode.Open));

        lblReader.ReadInt32();
        lblReader.ReadInt32();

        int magicNum = imgReader.ReadInt32();
        NumImgs = imgReader.ReadInt32();
        Rows = imgReader.ReadInt32();
        Cols = imgReader.ReadInt32();
        ImgDims = Rows * Cols;

        Debug.Log("Imgs: " + NumImgs + " Rows: " + Rows + " Cols: " + Cols);

        labels = new int[NumImgs];
        for (int i = 0; i < NumImgs; i++) {
            byte lbl = lblReader.ReadByte();
            labels[i] = (int)lbl;
        }

        pixels = new float[NumImgs * Rows * Cols];

        for (int i = 0; i < pixels.Length; i++) {
            byte pix = imgReader.ReadByte();
            pixels[i] = pix / 256f;
        }
    }

    // Todo: since all training data is in memory, and is readonly, why even copy it? Just return
    // a random subset of indices
    public static Batch GetBatch(int size, float[] pixels, int[] labels, System.Random r) {
        Batch b = new Batch(size);
        for (int i = 0; i < size; i++) {
            int randIndex = r.Next(Mnist.NumImgs);

            b.Labels[i] = labels[randIndex];
            for (int j = 0; j < ImgDims; j++) {
                b.Images[i][j] = pixels[randIndex * ImgDims + j];
            }
        }

        return b;
    }

    public static void ToTexture(float[] pixels, int imgIndex, Texture2D tex) {
        var colors = new Color[Rows * Cols];

        int firstPix = imgIndex * Rows * Cols;

        for (int y = 0; y < Cols; y++) {
            for (int x = 0; x < Rows; x++) {
                float pix = pixels[firstPix + y * Cols + x]; //  / 256f
                // Invert y
                colors[(Cols-1-y) * Cols + x] = new Color(pix, pix, pix, 1f);
            }
        }

        tex.SetPixels(0, 0, Rows, Cols, colors);
        tex.Apply(false);
    }

    public static void LabelToVector(int label, float[] vector) {
        for (int i = 0; i < 10; i++) {
            vector[i] = i == label ? 1f : 0f;
        }
    }

    public static void Subtract(float[] a, float[] b, float[] result) {
        if (a.Length != b.Length) {
            throw new System.ArgumentException("Lengths of arrays have to match");
        }

        for (int i = 0; i < a.Length; i++) {
            result[i] = a[i] - b[i];
        }
    }

    public static float Cost(float[] vector) {
        float sum = 0f;
        for (int i = 0; i < vector.Length; i++) {
            sum += Mathf.Abs(vector[i]);
        }
        return sum * sum;
    }
}

// From https://gist.github.com/Fuebar/7495914
class BigEndianBinaryReader : BinaryReader {
    public BigEndianBinaryReader(Stream input) : base(input) {

    }

    public override short ReadInt16() {
        byte[] b = ReadBytes(2);
        return (short)(b[1] + (b[0] << 8));
    }
    public override int ReadInt32() {
        byte[] b = ReadBytes(4);
        return b[3] + (b[2] << 8) + (b[1] << 16) + (b[0] << 24);
    }
    public override long ReadInt64() {
        byte[] b = ReadBytes(8);
        return b[7] + (b[6] << 8) + (b[5] << 16) + (b[4] << 24) + (b[3] << 32) + (b[2] << 40) + (b[1] << 48) + (b[0] << 56);
    }

    /// <summary>Returns <c>true</c> if the Int32 read is not zero, otherwise, <c>false</c>.</summary>
    /// <returns><c>true</c> if the Int32 is not zero, otherwise, <c>false</c>.</returns>
    public bool ReadInt32AsBool() {
        byte[] b = ReadBytes(4);
        if (b[0] == 0 || b[1] == 0 || b[2] == 0 || b[3] == 0)
            return false;
        else
            return true;
    }

    /// <summary>
    /// Reads a string prefixed by a 32-bit integer identifying its length, in chars.
    /// </summary>
    public string ReadString32BitPrefix() {
        int length = ReadInt32();
        return Encoding.ASCII.GetString(ReadBytes(length));
    }

    public float ReadFloat() {
        return (float)ReadDouble();
    }
}

public class Batch {
    public float[][] Images;
    public int[] Labels;

    public Batch(int size) {
        Images = new float[size][];
        for (int i = 0; i < size; i++) {
            Images[i] = new float[28*28];
        }
        Labels = new int[size];
    }
}