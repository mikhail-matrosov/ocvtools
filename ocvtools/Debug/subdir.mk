################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../AsyncCamera.cpp \
../LKSmooth.cpp \
../LKTracker.cpp 

OBJS += \
./AsyncCamera.o \
./LKSmooth.o \
./LKTracker.o 

CPP_DEPS += \
./AsyncCamera.d \
./LKSmooth.d \
./LKTracker.d 


# Each subdirectory must supply rules for building sources it contributes
%.o: ../%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/usr/local/include/opencv -O0 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@:%.o=%.d)" -o "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


