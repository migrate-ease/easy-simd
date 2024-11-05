
INSTALL_PATH=/usr/local/lib64/
WORK_PATH := .
CC=gcc

TARGET := add

# Src
SRCDIRS   := ${WORK_PATH}/

# C program
SRCEXTS   := .c .S

# Include easysimd
INCDIR += -I /usr/local/include/easysimd

# The flags
CFLAGS    := -O2 -Wall -DEASYSIMD_ENABLE_NATIVE_ALIASES
LDFLAGS   := 
LDFLAGS   := 
ARCH := $(shell uname -m)
ifeq ($(ARCH), aarch64)
	CFLAGS += -msve-vector-bits=128 -march=armv8.5-a+crc+sve2
	LDFLAGS += -msve-vector-bits=128 -march=armv8.5-a+crc+sve2
else
	CFLAGS += -march=native -msse 
	LDFLAGS += -march=native -msse 
endif

# The command used to delete file.
RM        = rm -f

SOURCES = $(foreach d,$(SRCDIRS),$(wildcard $(addprefix $(d)/*,$(SRCEXTS))))

OBJS    = $(foreach x,$(SRCEXTS), \
      $(patsubst %$(x), %.o, $(filter %$(x),$(SOURCES))))

.PHONY : all objs clean

all : $(TARGET)

# Rules for creating the dependency files (.d).
%.d : %.c
	$(CC) -MM -MD $(CFLAGS)  $<

# Rules for producing the objects.
objs : $(OBJS)
%.o : %.c
	@echo compiling $(notdir $<)
	$(SILENCE) $(CC) -c $(CFLAGS) $(INCDIR) $(LDFLAGS) -o $@ $<

%.o : %.S
	@echo compiling $(notdir $<)
	$(SILENCE) $(CC) -c $(CFLAGS) $(INCDIR) $(LDFLAGS) -o $@ $<

$(TARGET): $(OBJS)
	@echo Linking $@
	$(SILENCE) $(CC) $(CFLAGS) $(INCDIR) -o ./$(TARGET) $(OBJS) $(LDFLAGS)

clean :
	@-$(RM) *.d *.a *.so *.symbol $(TARGET) $(SOFTLINK).1 $(SOFTLINK)
	@-$(RM)
	@find ${WORK_PATH} -name '*.o' -exec $(RM) {} \;
	@echo all clean
