@Aspect
@Component
public class BeanUsageLoggerAspect {

    private static final Logger LOGGER = LoggerFactory.getLogger(BeanUsageLoggerAspect.class);

    @Before("execution(* com.yourpackage.YourBean.*(..))")
    public void logBeanUsage(JoinPoint joinPoint) {
        String beanName = joinPoint.getSignature().getDeclaringType().getSimpleName();
        LOGGER.info("Bean '{}' was used.", beanName);
    }
}
