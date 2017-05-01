package com.arahansa.neuralnet;

import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

import java.util.function.Function;

/**
 * Created by jarvis on 2017. 4. 27..
 */
@Component
public class J03_NumericGradient {

    /**
     * 기울기
     * @param f
     * @param x
     * @return
     */
    public Matrix getNumericGradient(Function<Matrix, Double> f, Matrix x){
        double h = 1e-4;

        Matrix grad = new Matrix(x.getShape(0), x.getShape(1));

        // j 가 x 좌표, i 가 y 좌표
        for (int i = 0; i < x.getShape(1); i++) {
            for (int j = 0; j < x.getShape(0); j++) {

                double tmp_val = x.get(j,i);

                // f(x+h) 계산
                x.set(j,i, (tmp_val+h) );
                double fxh1 = f.apply(x);


                // f(x-h) 계산
                x.set(j,i, (tmp_val-h) );
                double fxh2 = f.apply(x);

                grad.set(j,i, (fxh1 - fxh2) / (2 * h) );

                // 값 복원
                x.set(j,i, tmp_val);
            }
        }

        return grad;
    }
}
