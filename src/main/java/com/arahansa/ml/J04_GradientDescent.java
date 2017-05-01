package com.arahansa.ml;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.function.Function;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Slf4j
@Component
public class J04_GradientDescent {

    J03_NumericGradient numericGradient;

    @Autowired
    J04_GradientDescent(J03_NumericGradient j03_numericGradient){
        this.numericGradient = j03_numericGradient;
    }

    /**
     * 경사하강법
     * @param f
     * @param init_x
     * @param lr
     * @param step_num
     * @return
     */
    public Matrix getGradientDescent(Function<Matrix, Double> f, Matrix init_x, double lr, int step_num){
        Matrix x = init_x;

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
