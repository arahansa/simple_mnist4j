package com.arahansa.neuralnet;

import com.arahansa.data.Grad;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

import java.util.HashMap;
import java.util.Map;
import java.util.function.BiFunction;

import static com.arahansa.neuralnet.J00_Helper.multipleTwoMatrix;
import static com.arahansa.neuralnet.J00_Helper.sigmoid;
import static com.arahansa.neuralnet.J00_Helper.softmax;

/**
 * Created by arahansa on 2017-05-02.
 */
@Slf4j
@Component
public class J06_TwoLayerNet {

    Matrix W1, W2, b1, b2;

    /**
     * W1 ( input_size, hidden_size)
     *
     * @param input_size
     * @param hidden_size
     * @param output_size
     * @param weight_init_std
     */
    public J06_TwoLayerNet(int input_size, int hidden_size, int output_size, Float weight_init_std){
        if(weight_init_std==null) weight_init_std = 0.01f;

        W1 = J00_Helper.getRadNMatrix(input_size, hidden_size);
        W1.multiply(weight_init_std);
        b1= Matrix.create(1, hidden_size);

        W2 = J00_Helper.getRadNMatrix(hidden_size, output_size);
        W2.multiply(weight_init_std);
        b2= Matrix.create(1, output_size);
    }

    public Matrix predict(Matrix x){
        Matrix a1 = multipleTwoMatrix(x, W1);
        a1.add(b1.getRow(0));

        Matrix z1 = sigmoid(a1);
        Matrix a2 = multipleTwoMatrix(z1, W2);
        a2.add(b2.getRow(0));

        return softmax(a2);
    }

    BiFunction<Matrix, Matrix, Double> lossFunc = (x, t)->{
        Matrix y = predict(x);
        return J01_CostFunction.getCrossEntropyErr(y, t);
    };

    public double accuracy(Matrix x, Matrix t){
        Matrix y = predict(x);
        if(!t.isSameShape(y)){
            log.warn("t shape ({}, {}) , y shape ( {}, {} ) ", t.getShape(0), t.getShape(1), y.getShape(0), y.getShape(1));
            throw new IllegalStateException("different y , t size ");
        }
        log.info("y : {}", y);
        return calcul_accracy(y, t);
    }

    double calcul_accracy(Matrix y, Matrix t){
        int sameCount = 0;
        int size = y.getShape(0);
        for(int i = 0; i < size; i++){
            if(y.getRow(i).maxElementIndex()==t.getRow(i).maxElementIndex()){
                sameCount++;
            }
        }
        double accuracy = sameCount / new Double(size);
        return accuracy;
    }

    public Grad numerical_gradient(Matrix x, Matrix t){
        Grad grad = new Grad();
        Matrix w1 = J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.W1);

        grad.setW1(w1);
        grad.setB1(J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.b1));
        grad.setW2(J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.W2));
        grad.setB2(J07_Gradient.numerical_gradient(this.lossFunc, x, t, this.b2));

        return grad;
    }

    /**
     * 역전파 계산
     * @param x
     * @param t
     * @return
     */
    public Grad gradient(Matrix x, Matrix t){
        Grad grad = new Grad();
        int batch_num = x.getShape(0);

        // forward
        Matrix a1 = multipleTwoMatrix(x, this.W1);
        a1.add(this.b1.getRow(0));
        Matrix z1 = sigmoid(a1);
        Matrix a2 = multipleTwoMatrix(z1, this.W2);
        a2.add(this.b2.getRow(0));
        Matrix y = softmax(a2);

        // backward


        return grad;
    }

    public void renewParams(Grad grad, Double learning_rate){
        log.info("before renew param 1 : {}", this.W1.get(0,0));
        Matrix w1 = grad.getW1();
        w1.multiply(-1*learning_rate);
        log.info("grad w1 :{}", w1.get(0,0));
        W1.add(w1);
        log.info("before renew param 1 : {}", this.W1.get(0,0));

        Matrix w2 = grad.getW2();
        w2.multiply(-1*learning_rate);
        W2.add(w2);

        Matrix b1 = grad.getB1();
        b1.multiply(-1*learning_rate);
        this.b1.add(b1);

        Matrix b2 = grad.getB2();
        b2.multiply(-1*learning_rate);
        this.b2.add(b2);

    }





}
