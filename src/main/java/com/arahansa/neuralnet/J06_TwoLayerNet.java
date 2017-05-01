package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

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


    public J06_TwoLayerNet(int input_size, int hidden_size, int output_size, Float weight_init_std){
        if(weight_init_std==null) weight_init_std = 0.01f;

        W1 = J00_Helper.getRadNMatrix(input_size, hidden_size);
        W1.multiply(weight_init_std);
        b1= Matrix.create(1, hidden_size);

        W2 = J00_Helper.getRadNMatrix(input_size, hidden_size);
        W2.multiply(weight_init_std);
        b2= Matrix.create(1, output_size);
    }

    public Matrix predict(Matrix x){
        Matrix a1 = multipleTwoMatrix(x, W1);
        a1.add(b1);

        Matrix z1 = sigmoid(a1);
        Matrix a2 = multipleTwoMatrix(z1, W2);
        a2.add(b2);

        return softmax(a2);
    }

    /**
     * @param x : 입력 데이터
     * @param t : 정답 레이블
     * @return
     */
    public double loss(Matrix x, Matrix t){
        Matrix y = predict(x);
        return J01_CostFunction.getCrossEntropyErr(y, t);
    }

    // todo 
    public double accuracy(Matrix x, Matrix t){
        Matrix predict = predict(x);
        return predict.elementMax();
    }



}
