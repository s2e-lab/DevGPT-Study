@Autowired
private BeansEndpoint beansEndpoint;

public class Main {
    public void getClassAndPackageOfBeans() {
        Map<String, Object> beans = beansEndpoint.beans();

        for (Map.Entry<String, Object> entry : beans.entrySet()) {
            String beanName = entry.getKey();
            Object bean = entry.getValue();

            Class<?> beanClass = bean.getClass();
            String className = beanClass.getSimpleName();
            String packageName = beanClass.getPackageName();

            System.out.println("Bean Name: " + beanName);
            System.out.println("Class Name: " + className);
            System.out.println("Package Name: " + packageName);
            System.out.println("-----------------------------------");
        }
    }
}