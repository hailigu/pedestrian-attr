# use pkg-config for getting CFLAGS and LDLIBS
FFMPEG_LIBS=    libavdevice                        \
                libavformat                        \
                libavfilter                        \
                libavcodec                         \
                libswresample                      \
                libswscale                         \
                libavutil                          \




CFLAGS += -Wall -g -ldl -std=c99
CFLAGS := $(shell pkg-config --cflags $(FFMPEG_LIBS)) $(CFLAGS)
LDLIBS := $(shell pkg-config --libs $(FFMPEG_LIBS)) $(LDLIBS)

EXAMPLES=    demo 

SOURCES := $(shell find . -iname '*.c' )
OBJS := $(SOURCES:.c=.o)


# the following examples make explicit use of the math library
avcodec:           LDLIBS += -lm
decoding_encoding: LDLIBS += -lm
muxing:            LDLIBS += -lm
resampling_audio:  LDLIBS += -lm


.phony: all clean-test clean


all:  $(EXAMPLES) 

$(EXAMPLES): $(OBJS)
	gcc $(OBJS) $(CFLAGS) $(LDLIBS) -o $(EXAMPLES)

clean-test:
	$(RM) *.ts *.m3u8

clean: clean-test
	$(RM) $(EXAMPLES) $(OBJS)
