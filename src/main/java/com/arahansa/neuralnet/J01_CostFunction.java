package com.arahansa.neuralnet;

import lombok.extern.slf4j.Slf4j;
import mikera.matrixx.Matrix;
import org.springframework.stereotype.Component;

import java.util.function.BiFunction;

/**
 * Created by jarvis on 2017. 4. 27..
 * 교차 엔트로피 오차 구함
 */
@Slf4j
public class J01_CostFunction {

    /**
     * 교차 엔트로피 오차
     * -np.sum(t * np.log(y+delta))
     * @param y
     * @param t
     * @return
     */
    public static double getCrossEntropyErr(Matrix y, Matrix t){
        Matrix tClone = t.exactClone();
        double delta = 1e-7;
        y.add(delta);
        y.log();
        tClone.multiply(y);
        return -tClone.elementSum();
    }

    /**
     * 배치용 교차 엔트로피 오차 구하기
     * 파이썬 예제와는 다르게
     * 1, X 값으로 오는 것으로 기대하자.. 소프트맥스로처리 된 값을 받는다
     * 일단 안 쓸 듯하다
     * @param y
     * @param t
     * @return
     */
    public static double getCrossEntropyErr4Batch(Matrix y, Matrix t){
        // 1차원일 때의 처리는 생략. 왜냐면 매트릭스만 받는다..
        int batch_size = y.getShape(0);
        Matrix y_max = J00_Helper.collectByTmaxElem(y, t);
        y_max.log();
        double sum = y_max.elementSum();
        double v = sum / batch_size;
        return -v;
    }


}
