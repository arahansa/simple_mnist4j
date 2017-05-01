package com.arahansa;

import lombok.extern.slf4j.Slf4j;
import org.springframework.context.ApplicationContext;
import org.springframework.context.support.ClassPathXmlApplicationContext;

/**
 * Created by jarvis on 2017. 4. 26..
 */
@Slf4j
class MainApplication {


    ApplicationContext ctx;


    public static void main(String[] args) {
        MainApplication mainApplication = new MainApplication();
        mainApplication.ctx = new ClassPathXmlApplicationContext("");

    }

}
