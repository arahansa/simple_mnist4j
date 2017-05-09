package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.function.Function;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Slf4j
public class J04_GradientDescent {

    J03_NumericGradient numericGradient;

    J04_GradientDescent(J03_NumericGradient j03_numericGradient){
        this.numericGradient = j03_numericGradient;
    }

    /**
     * 경사하강법
     * @param f 기울기에 사용될 함수
     * @param init_x 초기 행렬
     * @param lr 학습률
     * @param step_num 반복수
     * @return
     */
    public Matrix getGradientDescent(Function<Matrix, Double> f,
                                     Matrix init_x,
                                     double lr, int step_num){

        Matrix x = init_x.copy();
        for(int i=0;i<step_num;i++){
            Matrix grad = numericGradient.getNumericGradient(f, x);

            log.debug("grad : {}", grad);
            grad.multiply(-1 * lr);

            x.add(grad);
            log.debug("x : {}", x);
        }
        return x;
    }

}
