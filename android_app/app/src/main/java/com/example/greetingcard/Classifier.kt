package com.example.greetingcard

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer

class Classifier(context: Context) {
    private var interpreter: Interpreter? = null
    private var inputImageWidth: Int = 0
    private var inputImageHeight: Int = 0

    // Define labels
    private val labels = listOf("Recyclable", "Not Recyclable")

    init {
        try {
            val model = FileUtil.loadMappedFile(context, "binary_classifier.tflite")
            val options = Interpreter.Options()
            interpreter = Interpreter(model, options)

            val inputTensor = interpreter!!.getInputTensor(0)
            val inputShape = inputTensor.shape() // {1, height, width, 3}
            inputImageHeight = inputShape[1]
            inputImageWidth = inputShape[2]
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    fun classify(bitmap: Bitmap): String {
        val tflite = interpreter ?: return "Model not initialized"

        val inputTensor = tflite.getInputTensor(0)
        val imageDataType = inputTensor.dataType()
        
        var tensorImage = TensorImage(imageDataType)
        tensorImage.load(bitmap)

        val imageProcessor = ImageProcessor.Builder()
            .add(ResizeOp(inputImageHeight, inputImageWidth, ResizeOp.ResizeMethod.BILINEAR))
            .build()
        
        tensorImage = imageProcessor.process(tensorImage)

        val outputTensor = tflite.getOutputTensor(0)
        val outputShape = outputTensor.shape() 
        val outputBuffer = TensorBuffer.createFixedSize(outputShape, outputTensor.dataType())
        
        tflite.run(tensorImage.buffer, outputBuffer.buffer.rewind())
        
        val scores = outputBuffer.floatArray
        
        // Find max score
        val maxIndex = scores.indices.maxByOrNull { scores[it] } ?: -1
        
        // Return corresponding label if index is valid, else "Unknown"
        return if (maxIndex in labels.indices) {
            labels[maxIndex]
        } else {
            "Unknown"
        }
    }

    fun close() {
        interpreter?.close()
    }
}
