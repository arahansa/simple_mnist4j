package com.arahansa.ml;

import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Slf4j
@Component
public class J05_SimpleNet {

    J01_CostFunction costFunction;
    J00_Helper helper;

    @Setter
    Matrix w;
    // 가지고 있으면서 loss  돌아가면서 계산될 녀석들
    @Setter
    Matrix x;
    @Setter
    Matrix t;


    @Autowired
    public J05_SimpleNet(J01_CostFunction costFunction, J00_Helper helper){
        this.helper = helper;
        this.costFunction = costFunction;
    }

    public Matrix predict(){
        Matrix newX = x;
        newX.multiply(w);
        return newX;
    }

    public double loss(){
        Matrix z = predict();
        Matrix y = helper.softmax(z);
        return costFunction.getCrossEntropyErr(y, t);
    }


    /**
     * (x = 초기 가중치)를 받고 다음 가중치를 보여줌
     * @param x
     * @return
     */
    public Matrix getSimpleNetDW(Matrix x){
        double h = 1e-4;
        Matrix grad = new Matrix(x.getShape(0), x.getShape(1));


        // j 가 x 좌표, i 가 y 좌표
        for (int i = 0; i < x.getShape(1); i++) {
            for (int j = 0; j < x.getShape(0); j++) {
                double tmp_val = x.get(j,i);


                // f(x+h) 계산
                x.set(j,i, (tmp_val+h) );
                double fxh1 = loss();


                // f(x-h) 계산
                x.set(j,i, (tmp_val-h) );
                double fxh2 = loss();

                grad.set(j,i, (fxh1 - fxh2) / (2 * h) );

                // 값 복원
                x.set(j,i, tmp_val);

            }
        }
        return grad;
    }



}
