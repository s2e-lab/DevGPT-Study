import java.util.*;
import java.util.stream.Collectors;

public class Main {
    public static void main(String[] args) {
        List<JarBean> jarBeans = new ArrayList<>();
        jarBeans.add(new JarBean("Jar1", "Bean1"));
        jarBeans.add(new JarBean("Jar1", "Bean2"));
        jarBeans.add(new JarBean("Jar2", "Bean1"));
        jarBeans.add(new JarBean("Jar2", "Bean3"));

        Map<String, Long> beanCountPerJar = jarBeans.stream()
                .collect(Collectors.groupingBy(JarBean::getJarName, Collectors.counting()));

        System.out.println(beanCountPerJar);
    }
}

class JarBean {
    private String jarName;
    private String beanName;

    public JarBean(String jarName, String beanName) {
        this.jarName = jarName;
        this.beanName = beanName;
    }

    public String getJarName() {
        return jarName;
    }

    public String getBeanName() {
        return beanName;
    }
}
