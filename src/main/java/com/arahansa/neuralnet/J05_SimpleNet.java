package com.arahansa.neuralnet;

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


    @Autowired
    public J05_SimpleNet(J00_Helper helper, J01_CostFunction costFunction){
        this.helper = helper;
        this.costFunction = costFunction;
    }

    public Matrix predict(Matrix x){
        return helper.multipleTwoMatrix(x, this.w);
    }

    public double loss(Matrix x, Matrix t){
        Matrix z = predict(x);
        Matrix y = helper.softmax(z);
        return costFunction.getCrossEntropyErr(y, t);
    }


    /**
     * (w = 초기 가중치)를 받고 다음 가중치를 보여줌
     * 책과 약간 다른 것은 책에서는 함수를 받도록 되어있으나 여기서는
     * loss 를 내부에서 가지고 있는다.
     * @param w
     * @return
     */
    public Matrix numerical_gradient(Matrix x, Matrix t, Matrix w){
        double h = 1e-4;
        Matrix grad = new Matrix(w.getShape(0), w.getShape(1));

        // i 가 x 좌표, j 가 y 좌표
        for (int i = 0; i < w.getShape(0); i++) {
            for (int j = 0; j < w.getShape(1); j++) {
                double tmp_val = w.get(i,j);
                log.info("현재 ({}, {}) 번째 값은 : {}", i, j, tmp_val);

                // f(x+h) 계산
                w.set(i,j, (tmp_val+h) );
                log.info(" 1 w : {}", w);
                this.setW(w);
                double fxh1 = loss(x, t);

                // f(x-h) 계산
                w.set(i,j, (tmp_val-h) );
                this.setW(w);
                log.info(" 2 w : {}", w);
                double fxh2 = loss(x, t);


                final double v = (fxh1 - fxh2) / (2 * h);
                log.info("loss1 : {}, loss2 : {}, 미분 : {}", fxh1, fxh2, v);
                grad.set(i,j, v);

                // 값 복원
                w.set(i,j, tmp_val);
                this.setW(w);
            }
        }
        return grad;
    }



}
